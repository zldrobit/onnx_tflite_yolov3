## Introduction
A Conversion tool to convert YOLO v3 Darknet weights to TF Lite model
(YOLO v3 in PyTorch > ONNX > TensorFlow > TF Lite).

## Prerequisites
- `python3`
- `torch==1.3.1`
- `torchvision==0.4.2`
- `onnx==1.6.0`
- `onnx-tf==1.5.0`
- `onnxruntime-gpu==1.0.0`
- `tensorflow-gpu==1.15.0`

## Docker
`docker pull zldrobit/onnx:10.0-cudnn7-devel`

## Usage
- **1. Download pretrained Darknet weights:**
```
cd weights
wget https://pjreddie.com/media/files/yolov3.weights 
```

- **2. Convert YOLO v3 model from Darknet weights to ONNX model:** 
Change `ONNX_EXPORT` to `True` in `models.py`. Run 
```
python3 detect.py --cfg cfg/yolov3.cfg --weights weights/yolov3.weights
```
The output ONNX file is `weights/export.onnx`. The input name is `input_1`. The output name is `output_1`.

- **3. (TensorRT) Convert ONNX model to TensorRT Engine file:**
```
# You'll need an nVidia NGC's account
docker pull nvcr.io/nvidia/tensorrt:20.03-py3
```
and run in docker:
```
bash run_trtexec.sh
```

## Auxiliary Files
- **ONNX inference and detection:** `onnx_infer.py` and `onnx_detect.py`.
- **TensorFlow inference and detection:** `tf_infer.py` and `tf_detect.py`.
- **TF Lite inference, detection and debug:** `tflite_infer.py`, `tflite_detect.py` 
and `tflite_debug.py`.

## Known Issues
- **The conversion code does not work with tensorflow==1.14.0:** Running prep.py cause protobuf error (Channel order issue in Conv2D).
- **fix_reshape.py does not fix shape attributes in TFLite tensors, which may cause unknown side effects.**

## TODO
- [x] **Add TensorRT support (see [onnx-tensorrt dynamic shape](https://github.com/onnx/onnx-tensorrt/issues/328) )**
- [ ] **Add TensorRT int8 calibration**
- [ ] **support conversion to TensorFlow model (related to [onnx-tensorflow Slice Op](https://github.com/onnx/onnx-tensorflow/issues/464))**

## Acknowledgement
We borrow PyTorch code from [ultralytics/yolov3](https://github.com/ultralytics/yolov3), 
and TensorFlow low-level API conversion code from [paulbauriegel/tensorflow-tools](https://github.com/paulbauriegel/tensorflow-tools).
[tensorrt-utils](https://github.com/rmccorm4/tensorrt-utils) is of great help in converting ONNX to TensorRT conversion.
  
