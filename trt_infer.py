import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # To automatically manage CUDA context creation and cleanup
import numpy as np
import time
import argparse


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
        print('size', size)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        print(size)
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


def main():
    parser = argparse.ArgumentParser(description="Benchmark TensorRT engine model.\n")
    parser.add_argument("--model", default="weights/model.fp16.batch1-64.opt32.nms.reshape.2G.engine", help="The engine model file to convert to benchmark")
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="The batch size for the TensorRT engine input")
    parser.add_argument("-n", "--ntrials", type=int, default=50, help="The number of benchmark trials")
    args, _ = parser.parse_known_args()
    

    engine_path = args.model
    height, width = 416, 416
    batch_size = args.batch_size

    TRT_LOGGER = trt.Logger()
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    with open(engine_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        print('has_implicit_batch_dimension:', engine.has_implicit_batch_dimension)
        print('binding 0 shape:', engine.get_binding_shape(0))
        print('binding 1 shape:', engine.get_binding_shape(1))
        print('max_batch_size', engine.max_batch_size)
        # engine.max_batch_size = batch_size

        # Allocate buffers and create a CUDA stream.
        inputs, outputs, dbindings, stream = allocate_buffers(engine, batch_size)

    with engine.create_execution_context() as context:
        img = np.random.randn(batch_size, 3, height, width)

        # 
        # Transfer input data to the GPU.
        np.copyto(inputs[0].host, np.reshape(img, [-1]))
        inp = inputs[0]
        cuda.memcpy_htod(inp.device, inp.host)

        # Set dynamic batch size.
        # context.setBindingDimensions([batch_size] + inp.shape[1:])
        print("all_binding_shapes_specified:", context.all_binding_shapes_specified)
        context.set_binding_shape(0, [batch_size, 3, height, width])
        print("all_binding_shapes_specified:", context.all_binding_shapes_specified)

        start = time.time()
        # Run inference.
        for i in range(args.ntrials):
            # context.execute(batch_size, dbindings)
            context.execute_v2(dbindings)

        end = time.time()
        print("ntrials:", args.ntrials, "dur:", end - start, "avg:", (end - start) / args.ntrials)

        out_ = outputs[0]
        # Transfer predictions back to host from GPU
        cuda.memcpy_dtoh(out_.host, out_.device)
        out_np = np.reshape(np.array(out_.host), [-1, 10647, 85])


if __name__ == "__main__":
    main()
