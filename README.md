## Introduction
A Conversion tool to convert YOLO v3 Darknet weights to TF Lite model
(YOLO v3 in PyTorch > ONNX > TensorFlow > TF Lite).

## Prerequisites
- `python3`
- `torch==1.5.0`
- `torchvision==0.6.0`
- `onnx==1.6.0`
- `onnx-tf==1.5.0`
- `onnxruntime-gpu==1.0.0`
- `tensorflow-gpu==1.15.0`

## Docker
```
docker pull zldrobit/onnx:10.0-cudnn7-devel-pytorch-1.5

# You'll need an nVidia NGC's account, run docker login before pulling
docker pull nvcr.io/nvidia/tensorrt:20.03-py3
```

## Usage
- **1. Download pretrained Darknet weights:**
```
cd weights
wget https://pjreddie.com/media/files/yolov3.weights 
```

- **2. Convert YOLO v3 model from Darknet weights to ONNX model (in ONNX docker):** 

Change `ONNX_EXPORT` and `TRT_NMS` to `True` in `models.py`. Run 
```
python3 detect.py --cfg cfg/yolov3.cfg --weights weights/yolov3.weights
```
The output ONNX file is `weights/export_trt_nms.onnx`. The input name is `input_1`. The output names are `boxes_1` and `scores_1`.

- **3. Build TensorRT batchedNMSPlugin with dynamic batch size support and overwrite the older plugin file (in TensorRT docker):**

Issue about batchedNMSPlugin with dynamic batch size: 
[Will BatchedNMS Plugin support runtime input dimensions (IPluginV2DynamicExt)?](https://github.com/NVIDIA/TensorRT/issues/544)
```
git clone https://github.com/zldrobit/TensorRT/tree/dynamic-BatchedNMS-mod
cd TensorRT
git submodule update --init --recursive
mkdir build
cd build
cmake ..
make nvinfer_plugin
cp libnvinfer_plugin.so.7.0.0 /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.7.0.0
```

- **4. Convert ONNX model to TensorRT Engine file (in TensorRT docker):**
```
./run_trt_convert.sh
```
The generated engine file is `weights/model.fp16.batch1-64.opt32.nms.reshape.2G.engine`

- **5. Predict with engine file (in TensorRT docker):**

Install opencv and matplotlib before prediction,
```
pip install opencv-python matplotlib
sudo apt install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
```
Then, run
`python3 trt_detect_nms.py`
or
`python3 trt_detect_nms_batch.py`

## Auxiliary Files
- **ONNX inference and detection:** `onnx_infer.py` and `onnx_detect.py`.
- **TensorFlow inference and detection:** `tf_infer.py` and `tf_detect.py`.
- **TF Lite inference, detection and debug:** `tflite_infer.py`, `tflite_detect.py` 
and `tflite_debug.py`.
- **TensorRT inference, detection and debug:** `trt_infer.py`, `trt_detect.py`, `trt_list_plugins.py`
and `trt_manual.py`.

## Known Issues
- **The conversion code does not work with tensorflow==1.14.0:** Running prep.py cause protobuf error (Channel order issue in Conv2D).
- **fix_reshape.py does not fix shape attributes in TFLite tensors, which may cause unknown side effects.**
- **PyTorch 1.3.1 crashes when exporting ONNX model in Step 2.**

## TODO
- [x] **Add TensorRT support (see [onnx-tensorrt dynamic shape](https://github.com/onnx/onnx-tensorrt/issues/328) )**
- [x] **Add TensorRT NMS support (see [Will BatchedNMS Plugin support runtime input dimensions (IPluginV2DynamicExt)?](https://github.com/NVIDIA/TensorRT/issues/544), or https://github.com/zldrobit/TensorRT/tree/dynamic-BatchedNMS-mod)**
- [ ] **Add TensorRT int8 calibration (see [onnx-tensorrt INT8 calibration](https://github.com/NVIDIA/TensorRT/issues/289))**
- [ ] **support conversion to TensorFlow model (related to [onnx-tensorflow Slice Op](https://github.com/onnx/onnx-tensorflow/issues/464))**

## Acknowledgement
We borrow PyTorch code from [ultralytics/yolov3](https://github.com/ultralytics/yolov3), 
and TensorFlow low-level API conversion code from [paulbauriegel/tensorflow-tools](https://github.com/paulbauriegel/tensorflow-tools).
[tensorrt-utils](https://github.com/rmccorm4/tensorrt-utils) is of great help in ONNX to TensorRT conversion.
  
