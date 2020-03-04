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
The output ONNX file is `weights/export.onnx`.

- **3. Convert ONNX model to TensorFlow model:**
```
python3 onnx2tf.py
``` 
The output file is `weights/yolov3.pb`.

- **4. Preprocess pb file to avoid NCHW conv, 5-D ops, and Int64 ops:**
```
python3 prep.py
``` 
The output file is `weights/yolov3_prep.pb`.

- **5. Use TOCO to convert pb -> tflite:**
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
Now, you can run `python3 tflite_detect.py` to detect objects in an image.

## Quantization
- **1. Install flatbuffers:**
Please refer to [flatbuffers](https://google.github.io/flatbuffers/flatbuffers_guide_building.html).

- **2. Download TFLite schema:**
```
wget https://github.com/tensorflow/tensorflow/raw/r1.15/tensorflow/lite/schema/schema.fbs
```

- **3. Run TOCO to convert and quantize pb -> tflite:**
```
toco --graph_def_file weights/yolov3_prep.pb \
    --output_file weights/yolov3_quant.tflite \
    --output_format TFLITE  \
    --input_arrays input.1 \
    --output_arrays concat_84 \
    --post_training_quantize
```
The output file is `weights/yolov3_quant.tflite`.

- **4. Convert tflite -> json:**
```
flatc -t --strict-json --defaults-json -o weights schema.fbs  -- weights/yolov3_quant.tflite
```
The output file is `weights/yolov3_quant.json`.

- **5. Fix ReshapeOptions:**
```
python3 fix_reshape.py
```
The output file is `weights/yolov3_quant_fix_reshape.json`.

- **6. Convert json -> tflite:**
```
flatc -b -o weights schema.fbs weights/yolov3_quant_fix_reshape.json
```
The output file is `weights/yolov3_quant_fix_reshape.tflite`.
Now, you can run 
```
python3 tflite_detect.py --weights weights/yolov3_quant_fix_reshape.tflite
``` 
to detect objects in an image.

## Auxiliary Files
- **ONNX inference and detection:** `onnx_infer.py` and `onnx_detect.py`.
- **TensorFlow inference and detection:** `tf_infer.py` and `tf_detect.py`.
- **TF Lite inference, detection and debug:** `tflite_infer.py`, `tflite_detect.py` 
and `tflite_debug.py`.

## Known Issues
- **The conversion code does not work with tensorflow==1.14.0:** Running prep.py cause protobuf error (Channel order issue in Conv2D).
- **fix_reshape.py does not fix shape attributes in TFLite tensors, which may cause unknown side effects.**

## TODO
- [x] **support quantized model**

## Acknowledgement
We borrow PyTorch code from [ultralytics/yolov3](https://github.com/ultralytics/yolov3), 
and TensorFlow low-level API conversion code from [paulbauriegel/tensorflow-tools](https://github.com/paulbauriegel/tensorflow-tools).
  
