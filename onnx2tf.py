import onnx 
from onnx_tf.backend import prepare  


onnx_model = onnx.load("weights/export.onnx")  # load onnx model 
# output = prepare(onnx_model).run(input)  # run the loaded model
# no strict to be faster
output = prepare(onnx_model, strict=False)

path = 'weights/yolov3.pb'

file = open(path, "wb")
file.write(output.graph.as_graph_def().SerializeToString())
file.close()
