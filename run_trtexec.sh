trtexec --onnx=weights/export.onnx --explicitBatch \
	--minShapes=input_1:1x3x416x416 \
	--optShapes=input_1:1x3x416x416 \
	--maxShapes=input_1:32x3x416x416 \
	--workspace=4096 \
	--fp16 \
	--saveEngine=weights/model.batch1-32.fp16.engine

# trtexec --explicitBatch --onnx=model.onnx --minShapes=input:1x3x224x224 --optShapes=input:32x3x224x224 --maxShapes=input:32x3x224x224 --saveEngine=model.batch1-32.engine
