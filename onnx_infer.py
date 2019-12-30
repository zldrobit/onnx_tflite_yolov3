import numpy as np
import onnxruntime as rt
import cv2
import time

height, width = 416, 416
n = 100

img = cv2.imread('data/samples/bus.jpg')
img = cv2.resize(img, (416, 416))
img = img[None, :, :, :]
img = np.transpose(img, [0, 3, 1, 2])

sess = rt.InferenceSession("weights/export.onnx")
input_name = sess.get_inputs()[0].name
start = time.time()
for i in range(n):
    pred_onx = sess.run(None, {input_name: img.astype(np.float32)})[0]
end = time.time()
print("Time: {} s, avg {} s".format(end - start, (end - start) / n))
