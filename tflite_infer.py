import numpy as np
import tensorflow as tf
import sys

# Load TFLite model and allocate tensors.
model_path = sys.argv[1] if len(sys.argv) > 1 else "weights/yolov3_quant.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(interpreter.get_tensor_details())

# Print details
all_layers_details = interpreter.get_tensor_details()
for layer in all_layers_details:
    print("index: {}, name: {}, shape: {}, quantization: {}, dtype: {}".format(
        layer['index'],
        layer['name'],
        layer['shape'],
        layer['quantization'],
        layer['dtype']))

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

# tensor = interpreter.get_tensor(7)
# print(tensor)
# print(tensor.name)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data.shape)
print(output_data)
