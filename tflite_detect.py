import argparse
from sys import platform
import sys

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

import cv2
import time

import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import load_delegate


def detect(save_txt=False, save_img=False):
    img_size = (416, 416)
    out, source, weights, half, view_img = opt.output, opt.source, opt.weights, opt.half, opt.view_img
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Load TFLite model and allocate tensors.
    if opt.delegate:
        interpreter = tf.lite.Interpreter(model_path=opt.weights, experimental_delegates=[load_delegate(opt.delegate)])
    else:
        interpreter = tf.lite.Interpreter(model_path=opt.weights)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

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

    OUTPUT_WIDTH = [52, 26, 13]
    NUM_BOXES_PER_BLOCK = 3
    INPUT_SIZE = 416 
    MASKS = [[0, 1, 2], [3, 4, 5], [6, 7, 8]];
    ANCHORS = [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326]

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        t = time.time()

        img0 = img
        start = time.time()
        for i in range(1):
            # Get detections
            # img = torch.from_numpy(img).to(device)
            # if img.ndimension() == 3:
            #     img = img.unsqueeze(0)
            # pred = model(img)[0]
            input_shape = input_details[0]['shape']
            print("input_shape", input_shape)

            img = img0
            img = img[None, :, :, :]
            img = np.transpose(img, [0, 2, 3, 1])
            input_data = img.astype(np.float32)
            print('input_data.shape', input_data.shape)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            out_ = []
            output_data0 = torch.Tensor(interpreter.get_tensor(output_details[0]['index']))
            output_data1 = torch.Tensor(interpreter.get_tensor(output_details[1]['index']))
            output_data2 = torch.Tensor(interpreter.get_tensor(output_details[2]['index']))
            out_.extend([output_data0[0], output_data1[0], output_data2[0]])
            output_data = torch.cat((
                output_data0.view(-1, 85),
                output_data1.view(-1, 85),
                output_data2.view(-1, 85)), 0)

            print('len(out)', len(out))

            for i in range(len(OUTPUT_WIDTH)):
                grid_width = OUTPUT_WIDTH[i]
                for y in range(grid_width):
                    for x in range(grid_width):
                        for b in range(NUM_BOXES_PER_BLOCK):
                            xPos = (x + torch.sigmoid(out_[i][y][x][b][0])) * (INPUT_SIZE / grid_width);
                            yPos = (y + torch.sigmoid(out_[i][y][x][b][1])) * (INPUT_SIZE / grid_width);
                            w = (torch.exp(out_[i][y][x][b][2]) * ANCHORS[2 * MASKS[i][b]]);
                            h = (torch.exp(out_[i][y][x][b][3]) * ANCHORS[2 * MASKS[i][b] + 1]);
                            out_[i][y][x][b][0] = xPos
                            out_[i][y][x][b][1] = yPos
                            out_[i][y][x][b][2] = w
                            out_[i][y][x][b][3] = h
            
            # print("output_data.shape", output_data.shape)
            # pred = torch.Tensor(output_data)
            pred = torch.cat((
                out_[0].view(-1, 85),
                out_[1].view(-1, 85),
                out_[2].view(-1, 85)), 0)
            pred = pred[None, ...]
            print('pred.shape', pred.shape)
            # pred = torch.Tensor(pred)
            if opt.half:
                pred = pred.float()

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)

            # Apply
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

        end = time.time()
        print("avg time:", (end - start) / 10)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            # print("len(det)", len(det))
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                det[:, :4] = scale_coords(img.shape[1:3], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, classes[int(c)])  # add to string

                # Write results
                for *xyxy, conf, _, cls in det:
                    print('xyxy', xyxy)
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
    parser.add_argument('--weights', type=str, default='weights/yolov3_coco.fp32.tflite', help='path to weights file')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--delegate', type=str, default='/workspace/tmp/libtensorflowlite_gpu_gl.so', help='TFLite delegate so file')  # output folder
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

