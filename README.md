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

## Usage
- **Download pretrained Darknet weights:**
```
cd weights
wget https://pjreddie.com/media/files/yolov3.weights 
```

- **Convert YOLO v3 model from Darknet weights to ONNX model:** 
Change `ONNX_EXPORT` to `True` in `models.py`. Run 
```
python3 detect.py --cfg cfg/yolov3.cfg --weights weights/yolov3.weights
```
The output ONNX file is `weights/export.onnx`.

- **Convert ONNX model to TensorFlow model:**
```
python3 onnx2tf.py
``` 
The output file is `weights/yolov3.pb`.

- **Preprocess pb file to avoid NCHW conv, 5-D ops, and Int64 ops:**
```
python3 prep.py
``` 
The output file is `weights/yolov3_prep.pb`.

- **Use TOCO to convert pb -> tflite:**
```
toco --graph_def_file weights/yolov3_prep.pb \
    --output_file weights/yolov3.tflite \
    --output_format TFLITE \
    --inference_type FLOAT \
    --inference_input_type FLOAT \
    --input_arrays input.1 \
    --output_arrays concat_84
```
The output file is `weights/yolov3.tflite`.
Now, you can run `tflite_detect.py` to detect objects in an image.

## Auxiliary Files
- **ONNX inference and detection:** `onnx_infer.py` and `onnx_detect.py`.
- **TensorFlow inference and detection:** `tf_infer.py` and `tf_detect.py`.
- **TF Lite inference, detection and debug:** `tflite_infer.py`, `tflite_detect.py` 
and `tflite_debug.py`.

## Known Issues
- **The conversion code does not work with tensorflow==1.14.0:** Running prep.py cause protobuf error (Channel order issue in Conv2D).

## Acknowledgement
We borrow PyTorch code from [ultralytics/yolov3](https://github.com/ultralytics/yolov3), 
and TensorFlow low-level API conversion code from [paulbauriegel/tensorflow-tools](https://github.com/paulbauriegel/tensorflow-tools).
  
