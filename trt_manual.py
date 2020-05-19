import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # To automatically manage CUDA context creation and cleanup
import numpy as np
import logging
import time

TRT_LOGGER = trt.Logger()

trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def add_profiles(config, inputs, opt_profiles):
    logger.debug("=== Optimization Profiles ===")
    for i, profile in enumerate(opt_profiles):
        for inp in inputs:
            _min, _opt, _max = profile.get_shape(inp.name)
            logger.debug("{} - OptProfile {} - Min {} Opt {} Max {}".format(inp.name, i, _min, _opt, _max))
        config.add_optimization_profile(profile)

def mark_outputs(network):
    # Mark last layer's outputs if not already marked
    # NOTE: This may not be correct in all cases
    last_layer = network.get_layer(network.num_layers-1)
    if not last_layer.num_outputs:
        logger.error("Last layer contains no outputs.")
        return

    for i in range(last_layer.num_outputs):
        network.mark_output(last_layer.get_output(i))


def check_network(network):
    if not network.num_outputs:
        logger.warning("No output nodes found, marking last layer's outputs as network outputs. Correct this if wrong.")
        mark_outputs(network)
    
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    max_len = max([len(inp.name) for inp in inputs] + [len(out.name) for out in outputs])

    logger.debug("=== Network Description ===")
    for i, inp in enumerate(inputs):
        logger.debug("Input  {0} | Name: {1:{2}} | Shape: {3}".format(i, inp.name, max_len, inp.shape))
    for i, out in enumerate(outputs):
        logger.debug("Output {0} | Name: {1:{2}} | Shape: {3}".format(i, out.name, max_len, out.shape))


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


def get_trt_plugin(plugin_name):
        plugin = None
        for plugin_creator in PLUGIN_CREATORS:
            if plugin_creator.name == plugin_name:
                print('plugin_creator.plugin_namespace:', plugin_creator.plugin_namespace)
                # lrelu_slope_field = trt.PluginField("neg_slope", np.array([0.1], dtype=np.float32), trt.PluginFieldType.FLOAT32)
                bnms_share_location_field = trt.PluginField("shareLocation", np.array([share_location], dtype=np.bool), trt.PluginFieldType.INT32)
                bnms_background_label_id_field = trt.PluginField("backgroundLabelId", np.array([-1], dtype=np.int32), trt.PluginFieldType.INT32)
                bnms_num_classes_field = trt.PluginField("numClasses", np.array([number_classes], dtype=np.int32), trt.PluginFieldType.INT32)
                bnms_topk_field = trt.PluginField("topK", np.array([topk], dtype=np.int32), trt.PluginFieldType.INT32)
                bnms_keep_topk_field = trt.PluginField("keepTopK", np.array([keep_topk], dtype=np.int32), trt.PluginFieldType.INT32)
                bnms_score_threshold_field = trt.PluginField("scoreThreshold", np.array([0.5], dtype=np.float32), trt.PluginFieldType.FLOAT32)
                bnms_iou_threshold_field = trt.PluginField("iouThreshold", np.array([0.5], dtype=np.float32), trt.PluginFieldType.FLOAT32)
                bnms_is_normalized_field = trt.PluginField("isNormalized", np.array([is_normalized], dtype=np.bool), trt.PluginFieldType.INT32)
                bnms_clip_boxes_field = trt.PluginField("clipBoxes", np.array([clip_boxes], dtype=np.bool), trt.PluginFieldType.INT32)
                field_collection = trt.PluginFieldCollection([
                    bnms_share_location_field,
                    bnms_background_label_id_field,
                    bnms_num_classes_field,
                    bnms_topk_field,
                    bnms_keep_topk_field,
                    bnms_score_threshold_field,
                    bnms_iou_threshold_field,
                    bnms_is_normalized_field,
                ])
                plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)
        return plugin

batch_size = 64
number_boxes = 3000
number_classes = 80
share_location = True
is_normalized = True
topk = 1000
keep_topk = 100
clip_boxes = False
ntrials = int(1)

with trt.Builder(TRT_LOGGER) as builder, \
     builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
     builder.create_builder_config() as config:

    print('has_implicit_batch_dimension', network.has_implicit_batch_dimension)

    builder.max_batch_size = batch_size
    config.max_workspace_size = 2**32
    # config.max_batch_size = batch_size
    q = 1 if share_location else number_classes
    input0 = network.add_input(name="input0", dtype=trt.float32, shape=(batch_size, number_boxes, q, 4))
    input1 = network.add_input(name="input1", dtype=trt.float32, shape=(batch_size, number_boxes, number_classes))
    bnms = network.add_plugin_v2(inputs=[input0, input1], plugin=get_trt_plugin("BatchedNMS_TRT"))
    # bnms.get_output(0).name = "outputs"
    # network.mark_output(bnms.get_output(0))
    # mark_outputs(network)
    bnms.get_output(0).name = "num_detections"
    network.mark_output(bnms.get_output(0))
    bnms.get_output(1).name = "nmsed_boxes"
    network.mark_output(bnms.get_output(1))
    bnms.get_output(2).name = "nmsed_scores"
    network.mark_output(bnms.get_output(2))
    bnms.get_output(3).name = "nmsed_classes"

    # import pdb
    # pdb.set_trace()
    network.mark_output(bnms.get_output(3))
    check_network(network)
    engine = builder.build_engine(network, config)
    for binding in engine:
        print(engine.get_binding_shape(binding), engine.get_binding_dtype(binding))
    inputs, outputs, dbindings, stream = allocate_buffers(engine, batch_size)

with engine.create_execution_context() as context:
    box = np.array([10.0, 20.0, 99.0, 200.0] * number_boxes * q * batch_size)
    print('box.shape', box.shape)
    np.copyto(inputs[0].host, np.reshape(box, [-1]))
    scores = np.array([0.9] * number_classes * number_boxes * batch_size)
    print('scores.shape', scores.shape)
    np.copyto(inputs[1].host, np.reshape(scores, [-1]))
    cuda.memcpy_htod(inputs[0].device, inputs[0].host)
    cuda.memcpy_htod(inputs[1].device, inputs[1].host)
    # context.set_binding_shape(0, [batch_size, 3, height, width])
    start = time.time()
    for i in range(ntrials):
        context.execute(batch_size, dbindings)
    end = time.time()
    print("dur:", end - start, "avg:", (end - start) / ntrials)
    for oup in outputs:
        cuda.memcpy_dtoh(oup.host, oup.device)
        print(oup.host)
    

