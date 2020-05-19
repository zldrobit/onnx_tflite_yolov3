import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

import cv2
import time

import os
import glob
import argparse
import PIL.Image
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # To automatically manage CUDA context creation and cleanup


class HostDeviceMem(object):
    r""" Simple helper data class that's a little nicer to use than a 2-tuple.
    """
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine: trt.ICudaEngine, batch_size: int):
    print('Allocating buffers ...')

    inputs = []
    outputs = []
    dbindings = []

    stream = cuda.Stream()

    for binding in engine:
        # size = batch_size * trt.volume(-1 * engine.get_binding_shape(binding))
        size = batch_size * trt.volume(engine.get_binding_shape(binding)[1:])
        # print('size', size)
        print('shape', engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        print('dtype', dtype)
        # Allocate host and device buffers
        # print(size)
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        dbindings.append(int(device_mem))

        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, dbindings, stream


def detect(save_txt=False, save_img=False):
    engine_path = "weights/model.fp16.batch1-64.opt32.nms.reshape.2G.engine"
    batch_size = 1

    # img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    img_size = (416, 416)
    out, source, weights, half, view_img = opt.output, opt.source, opt.weights, opt.half, opt.view_img
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size, half=half)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size, half=half)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(opt.data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    with open(engine_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        TRT_LOGGER = trt.Logger()
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        engine = runtime.deserialize_cuda_engine(f.read())

        # Allocate buffers and create a CUDA stream.
        inputs, outputs, dbindings, stream = allocate_buffers(engine, batch_size)


    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        # Contexts are used to perform inference.
        with engine.create_execution_context() as context:
            t = time.time()

            img0 = img
            start = time.time()
            for i in range(1):
                # Get detections
                # img = torch.from_numpy(img).to(device)
                # if img.ndimension() == 3:
                #     img = img.unsqueeze(0)
                # pred = model(img)[0]
                img = img0
                img = img[None, :, :, :]

                # test_images = np.random.choice(input_images, size=batch_size)
                # load_normalized_test_case(test_images, inputs[0].host, preprocess_func)

                np.copyto(inputs[0].host, np.reshape(img, [-1]))

                inp = inputs[0]
                # Transfer input data to the GPU.
                cuda.memcpy_htod(inp.device, inp.host)

                # import pdb
                # pdb.set_trace()

                # Set dynamic batch size.
                # context.setBindingDimensions([batch_size] + inp.shape[1:])
                context.set_binding_shape(0, [batch_size, 3, 416, 416])

                # Run inference.
                context.execute_v2(dbindings)

                # Transfer predictions back to host from GPU
                keep_topk = 100
                num_classes = 80
                cuda.memcpy_dtoh(outputs[0].host, outputs[0].device)
                num_detections = np.reshape(np.array(outputs[0].host), [])
                cuda.memcpy_dtoh(outputs[1].host, outputs[1].device)
                nmsed_boxes = np.reshape(np.array(outputs[1].host), [batch_size, keep_topk, 4])
                cuda.memcpy_dtoh(outputs[2].host, outputs[2].device)
                nmsed_scores = np.reshape(np.array(outputs[2].host), [batch_size, keep_topk])
                cuda.memcpy_dtoh(outputs[3].host, outputs[3].device)
                nmsed_classes = np.reshape(np.array(outputs[3].host), [batch_size, keep_topk])

                print('num_detections', num_detections)

                # assume batch_size = 1
                ndets = num_detections
                classes_ = nmsed_classes[0, :ndets]
                classes_ = torch.Tensor(classes_)
                probs = nmsed_scores[0, :ndets]
                print('probs', probs)
                boxes = nmsed_boxes[0, :ndets, :]
                print('boxes', boxes)
                print('boxes.dtype', boxes.dtype)
                # boxes = xywh2xyxy(boxes)
                boxes = torch.Tensor(boxes)

        end = time.time()
        print("avg time:", (end - start) / 100)

        # Process detections
        p, s, im0 = path, '', im0s

        save_path = str(Path(out) / Path(p).name)
        s += '%gx%g ' % img.shape[2:]  # print string
        if ndets > 0:
            # Rescale boxes from img_size to im0 size
            boxes = scale_coords(img.shape[2:], boxes, im0.shape).round()

            # Print results
            for c in classes_.unique():
                n = (classes_== c).sum()  # detections per class
                s += '%g %ss, ' % (n, classes[int(c)])  # add to string

            # Write results
            # for *xyxy, conf, _, cls in det:

            for i in range(ndets):
                xyxy = boxes[i]
                print("xyxy", xyxy)
                conf = probs[i]
                cls = classes_[i]
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                if save_img or view_img:  # Add bbox to image
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

        print('%sDone. (%.3fs)' % (s, time.time() - t))

        # Stream results
        if view_img:
            cv2.imshow(p, im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'images':
                cv2.imwrite(save_path, im0)
            else:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
