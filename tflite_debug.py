from tflite import Model
import pdb


buf = open('weights/int32.tflite', 'rb').read()
model = Model.GetRootAsModel(buf, 0)
subgraph = model.Subgraphs(0)
print(dir(subgraph))
print(subgraph.OperatorsLength())

n_ops = subgraph.OperatorsLength() 
# n_adds = 0
# for i_op in range(n_ops):
#     op = subgraph.Operators(i_op)
#     # Opcode for ADD
#     if op.OpcodeIndex() == 0:
#         print(i_op)
#         n_adds += 1
# for i_op in range(n_ops):
#     op = subgraph.Operators(i_op)

i_op = 439
op = subgraph.Operators(i_op)
pdb.set_trace()

for i_op in range(n_ops):
    op = subgraph.Operators(i_op)

    # print(op.InputsAsNumpy())
    # print(op.OutputsAsNumpy())

    # i_in1, i_in2 = op.InputsAsNumpy()
    i_out1 = op.OutputsAsNumpy()[0]

    # i_in1 = 69
    # in1 = subgraph.Tensors(i_in1)
    # print(dir(in1))
    # print(in1.Type())

    # i_in2 = 269
    # in2 = subgraph.Tensors(i_in2)
    # print(in2.Type())

    # i_out1 = 287
    out1 = subgraph.Tensors(i_out1)
    # print(out1.Type())
    if out1.Type() != 0:
        print(op.OpcodeIndex(), out1.Type(), i_out1)

	
# Check tensor.Name() to find the tensor_idx you want
# tensor = subgraph.Tensors(0) 
# buffer_idx = tensor.Buffer()
# buffer = model.Buffers(buffer_idx)
