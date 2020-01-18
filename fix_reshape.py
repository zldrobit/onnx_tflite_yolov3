import json

model = json.load(open('weights/yolov3_quant.json'))
ops = model['subgraphs'][0]['operators']
for op in ops:
    if op['builtin_options_type'] == 'ReshapeOptions':
        op['builtin_options']['new_shape'] = list()

json.dump(model, open('weights/yolov3_quant_fix_reshape.json', 'w'))
