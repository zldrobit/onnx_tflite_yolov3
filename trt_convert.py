#!/usr/bin/env python3

# Copyright 2019 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import glob
import math
import logging
import argparse
import types
import numpy as np

import tensorrt as trt

# from ImagenetCalibrator import ImagenetCalibrator, get_calibration_files, get_int8_calibrator # local module

TRT_LOGGER = trt.Logger()
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


def get_batch_sizes(max_batch_size):
    # Returns powers of 2, up to and including max_batch_size
    max_exponent = math.log2(max_batch_size)
    for i in range(int(max_exponent)+1):
        batch_size = 2**i
        yield batch_size
    
    if max_batch_size != batch_size:
        yield max_batch_size


# TODO: This only covers dynamic shape for batch size, not dynamic shape for other dimensions
def create_optimization_profiles(builder, inputs, batch_sizes=[1,8,16,32,64]): 
    # Check if all inputs are fixed explicit batch to create a single profile and avoid duplicates
    if all([inp.shape[0] > -1 for inp in inputs]):
        profile = builder.create_optimization_profile()
        for inp in inputs:
            fbs, shape = inp.shape[0], inp.shape[1:]
            # profile.set_shape(inp.name, min=(fbs, *shape), opt=(fbs, *shape), max=(fbs, *shape))
            profile.set_shape(inp.name, min=(1, *shape), opt=(fbs, *shape), max=(fbs, *shape))
            return [profile]
    
    # Otherwise for mixed fixed+dynamic explicit batch inputs, create several profiles
    profiles = {}
    for bs in batch_sizes:
        if not profiles.get(bs):
            profiles[bs] = builder.create_optimization_profile()

        for inp in inputs: 
            shape = inp.shape[1:]
            # Check if fixed explicit batch
            if inp.shape[0] > -1:
                bs = inp.shape[0]

            # profiles[bs].set_shape(inp.name, min=(bs, *shape), opt=(bs, *shape), max=(bs, *shape))
            profiles[bs].set_shape(inp.name, min=(1, *shape), opt=(32, *shape), max=(bs, *shape))

    return list(profiles.values())


def get_trt_plugin(plugin_name, plugin_args):
        plugin = None
        for plugin_creator in PLUGIN_CREATORS:
            if plugin_creator.name == plugin_name:
                print('plugin_creator.plugin_namespace:', plugin_creator.plugin_namespace)
                # lrelu_slope_field = trt.PluginField("neg_slope", np.array([0.1], dtype=np.float32), trt.PluginFieldType.FLOAT32)
                bnms_share_location_field = trt.PluginField("shareLocation", np.array([plugin_args.share_location], dtype=np.bool), trt.PluginFieldType.INT32)
                bnms_background_label_id_field = trt.PluginField("backgroundLabelId", np.array([plugin_args.background_label_id], dtype=np.int32), trt.PluginFieldType.INT32)
                bnms_num_classes_field = trt.PluginField("numClasses", np.array([plugin_args.number_classes], dtype=np.int32), trt.PluginFieldType.INT32)
                bnms_topk_field = trt.PluginField("topK", np.array([plugin_args.topk], dtype=np.int32), trt.PluginFieldType.INT32)
                bnms_keep_topk_field = trt.PluginField("keepTopK", np.array([plugin_args.keep_topk], dtype=np.int32), trt.PluginFieldType.INT32)
                bnms_score_threshold_field = trt.PluginField("scoreThreshold", np.array([plugin_args.score_threshold], dtype=np.float32), trt.PluginFieldType.FLOAT32)
                bnms_iou_threshold_field = trt.PluginField("iouThreshold", np.array([plugin_args.iou_threshold], dtype=np.float32), trt.PluginFieldType.FLOAT32)
                bnms_is_normalized_field = trt.PluginField("isNormalized", np.array([plugin_args.is_normalized], dtype=np.bool), trt.PluginFieldType.INT32)
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


def main():
    parser = argparse.ArgumentParser(description="Creates a TensorRT engine from the provided ONNX file.\n")
    parser.add_argument("--onnx", required=True, help="The ONNX model file to convert to TensorRT")
    parser.add_argument("-o", "--output", type=str, default="weights/model.engine", help="The path at which to write the engine")
    parser.add_argument("-b", "--max-batch-size", type=int, default=32, help="The max batch size for the TensorRT engine input")
    parser.add_argument("-v", "--verbosity", action="count", help="Verbosity for logging. (None) for ERROR, (-v) for INFO/WARNING/ERROR, (-vv) for VERBOSE.")
    parser.add_argument("--explicit-batch", action='store_true', help="Set trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH.")
    parser.add_argument("--explicit-precision", action='store_true', help="Set trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION.")
    parser.add_argument("--gpu-fallback", action='store_true', help="Set trt.BuilderFlag.GPU_FALLBACK.")
    parser.add_argument("--refittable", action='store_true', help="Set trt.BuilderFlag.REFIT.")
    parser.add_argument("--debug", action='store_true', help="Set trt.BuilderFlag.DEBUG.")
    parser.add_argument("--strict-types", action='store_true', help="Set trt.BuilderFlag.STRICT_TYPES.")
    parser.add_argument("--fp16", action="store_true", help="Attempt to use FP16 kernels when possible.")
    parser.add_argument("--int8", action="store_true", help="Attempt to use INT8 kernels when possible. This should generally be used in addition to the --fp16 flag. \
                                                             ONLY SUPPORTS RESNET-LIKE MODELS SUCH AS RESNET50/VGG16/INCEPTION/etc.")
    parser.add_argument("--nms", action="store_true", help="Proceed NMS in TensorRT.")
    parser.add_argument("--calibration-cache", help="(INT8 ONLY) The path to read/write from calibration cache.", default="calibration.cache")
    parser.add_argument("--calibration-data", help="(INT8 ONLY) The directory containing {*.jpg, *.jpeg, *.png} files to use for calibration. (ex: Imagenet Validation Set)", default=None)
    parser.add_argument("--calibration-batch-size", help="(INT8 ONLY) The batch size to use during calibration.", type=int, default=32)
    parser.add_argument("--max-calibration-size", help="(INT8 ONLY) The max number of data to calibrate on from --calibration-data.", type=int, default=512)
    parser.add_argument("-p", "--preprocess_func", type=str, default=None, help="(INT8 ONLY) Function defined in 'processing.py' to use for pre-processing calibration data.")
    args, _ = parser.parse_known_args()

    # Adjust logging verbosity
    if args.verbosity is None:
        TRT_LOGGER.min_severity = trt.Logger.Severity.ERROR
    # -v
    elif args.verbosity == 1:
        TRT_LOGGER.min_severity = trt.Logger.Severity.INFO
    # -vv
    else:
        TRT_LOGGER.min_severity = trt.Logger.Severity.VERBOSE
    logger.info("TRT_LOGGER Verbosity: {:}".format(TRT_LOGGER.min_severity))

    # Network flags
    network_flags = 0
    if args.explicit_batch:
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if args.explicit_precision:
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)

    builder_flag_map = {
            'gpu_fallback': trt.BuilderFlag.GPU_FALLBACK,
            'refittable': trt.BuilderFlag.REFIT,
            'debug': trt.BuilderFlag.DEBUG,
            'strict_types': trt.BuilderFlag.STRICT_TYPES,
            'fp16': trt.BuilderFlag.FP16,
            'int8': trt.BuilderFlag.INT8,
    }

    # Building engine
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(network_flags) as network, \
         builder.create_builder_config() as config, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
            
        print('has_implicit_batch_dimension', network.has_implicit_batch_dimension)
        config.max_workspace_size = 2**31 # 2GiB

        # Set Builder Config Flags
        for flag in builder_flag_map:
            if getattr(args, flag):
                logger.info("Setting {}".format(builder_flag_map[flag]))
                config.set_flag(builder_flag_map[flag])

        if args.fp16 and not builder.platform_has_fast_fp16:
            logger.warning("FP16 not supported on this platform.")

        if args.int8 and not builder.platform_has_fast_int8:
            logger.warning("INT8 not supported on this platform.")

        if args.int8:
            config.int8_calibrator = get_int8_calibrator(args.calibration_cache,
                                                         args.calibration_data,
                                                         args.max_calibration_size,
                                                         args.preprocess_func,
                                                         args.calibration_batch_size)

        # Fill network atrributes with information by parsing model
        with open(args.onnx, "rb") as f:
            if not parser.parse(f.read()):
                print('ERROR: Failed to parse the ONNX file: {}'.format(args.onnx))
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                sys.exit(1)

        # Insert NMS layer
        if args.nms:
            trt.init_libnvinfer_plugins(TRT_LOGGER, '')
            global PLUGIN_CREATORS
            PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list
            # see https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin 
            # for plugin arguments 
            plugin_args = types.SimpleNamespace()
            plugin_args.share_location = 1
            plugin_args.background_label_id = -1
            plugin_args.number_classes = 80
            plugin_args.topk = 1000
            plugin_args.keep_topk = 100
            plugin_args.score_threshold = 0.3
            plugin_args.iou_threshold = 0.5
            plugin_args.is_normalized = 0
            boxes_1 = network.get_output(0)
            print('boxes_1.shape', boxes_1.shape)
            # see https://github.com/NVIDIA/TensorRT/issues/166 for TensorRT reshape op
            boxes_reshape_op = network.add_shuffle(input=boxes_1)
            boxes_reshape_op.reshape_dims = [-1, 10647, 1, 4]
            boxes_reshape = boxes_reshape_op.get_output(0)
            print('boxes_reshape.shape', boxes_reshape.shape)

            scores_1 = network.get_output(1)
            print('scores_1.shape', scores_1.shape)
            scores_reshape_op = network.add_shuffle(input=scores_1)
            scores_reshape_op.reshape_dims = [-1, 10647, 80]
            scores_reshape = scores_reshape_op.get_output(0)
            print('scores_reshape.shape', scores_reshape.shape)

            nms = network.add_plugin_v2(
                    inputs=[boxes_reshape, scores_reshape], 
                    plugin=get_trt_plugin("BatchedNMS_TRT", plugin_args))
            print('nms.plugin.plugin_namespace', nms.plugin.plugin_namespace)
            nms.plugin.plugin_namespace = ""
            print('nms.plugin.plugin_namespace', nms.plugin.plugin_namespace)

            nms.get_output(0).name = "num_detections"
            print("nms.get_output(0).shape", nms.get_output(0).shape)
            # reshape num_detections to adapt dynamic batch dim of 
            # Triton (TensorRT) Inference Server
            det = nms.get_output(0)
            det_reshape_op = network.add_shuffle(input=det)
            det_reshape_op.reshape_dims = [-1, 1]
            det_reshape = det_reshape_op.get_output(0)
            print("det_rehshape.shape", det_reshape.shape)
            det_reshape.name = "num_detections_reshaped"
            network.mark_output(det_reshape)

            # network.mark_output(nms.get_output(0))
            nms.get_output(1).name = "nmsed_boxes"
            network.mark_output(nms.get_output(1))
            nms.get_output(2).name = "nmsed_scores"
            network.mark_output(nms.get_output(2))
            nms.get_output(3).name = "nmsed_classes"
            network.mark_output(nms.get_output(3))
            # print(boxes, boxes.name)
            # print(scores, scores.name)
            network.unmark_output(boxes_1)
            network.unmark_output(scores_1)

        # Display network info and check certain properties
        check_network(network)

        if args.explicit_batch:
            # Add optimization profiles
            # batch_sizes = [1, 8, 16, 32, 64]
            batch_sizes = [64]
            inputs = [network.get_input(i) for i in range(network.num_inputs)]
            opt_profiles = create_optimization_profiles(builder, inputs, batch_sizes)
            add_profiles(config, inputs, opt_profiles)
        # Implicit Batch Network
        else:
            builder.max_batch_size = args.max_batch_size

        logger.info("Building Engine...")
        with builder.build_engine(network, config) as engine, open(args.output, "wb") as f:
            logger.info("Serializing engine to file: {:}".format(args.output))
            f.write(engine.serialize())

if __name__ == "__main__":
    main()
