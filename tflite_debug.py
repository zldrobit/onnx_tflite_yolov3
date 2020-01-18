import tflite
from tflite import Model
import pdb


buf = open('weights/yolov3_quant.tflite', 'rb').read()
model = Model.GetRootAsModel(buf, 0)
subgraph = model.Subgraphs(0)
# print(dir(subgraph))
print('subgraph.OperatorsLength()', subgraph.OperatorsLength())

n_ops = subgraph.OperatorsLength() 
print('n_ops', n_ops)
i_op = 0
# op = subgraph.Operators(i_op)
# print("Inputs", op.Inputs(0), op.Inputs(1))
# print(op.BuiltinOptionsType())
# assert(op.BuiltinOptionsType() == tflite.BuiltinOptions.ReshapeOptions)
for i_op in range(n_ops):
    op = subgraph.Operators(i_op)
    if op.BuiltinOptionsType() == tflite.BuiltinOptions.ReshapeOptions:
        print(i_op)
        i_out = op.Outputs(0)
        out = subgraph.Tensors(i_out) 
        print('out', out.Name())
        op_opt = op.BuiltinOptions()
        opt = tflite.ReshapeOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        print(opt.NewShapeAsNumpy())

# print(op.BuiltinOptions())
# print(0, "Offset", table.Offset(0))
# print(dir(op))
# print(1, "Offset", table.Offset(1))
# print(2, "Offset", table.Offset(2))
# print(3, "Offset", table.Offset(3))
# print(dir(op))

# print(tensor.Name())

