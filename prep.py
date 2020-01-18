import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.core.framework import types_pb2
graph_def_file = "weights/yolov3.pb"

import pdb

tf.reset_default_graph()
graph_def = tf.GraphDef()
del_nodes = []
with tf.Session() as sess:
    # Read binary pb graph from file
    with tf.gfile.Open(graph_def_file, "rb") as f:
        data2read = f.read()
        graph_def.ParseFromString(data2read)
    tf.graph_util.import_graph_def(graph_def, name='')
    
    # Get Nodes
    conv_nodes = [n for n in sess.graph.get_operations() if n.type in ['Conv2D','MaxPool','AvgPool']]
    for n_org in conv_nodes:
        # print(n_org.name, n_org.type) 
        # Transpose input
        assert len(n_org.inputs)==1 or len(n_org.inputs)==2
        org_inp_tens = sess.graph.get_tensor_by_name(n_org.inputs[0].name)
        inp_tens = tf.transpose(org_inp_tens, [0, 2, 3, 1], name=n_org.name +'_transp_input')
        op_inputs = [inp_tens]
        
        # Get filters for Conv but don't transpose
        if n_org.type == 'Conv2D':
            filter_tens = sess.graph.get_tensor_by_name(n_org.inputs[1].name)
            op_inputs.append(filter_tens)
        
        # Attributes without data_format, NWHC is default
        atts = {key:n_org.node_def.attr[key] for key in list(n_org.node_def.attr.keys()) if key != 'data_format'}
        if 'ksize' in atts:
            kl = atts['ksize'].list.i
            ksl = [kl[0], kl[2], kl[3], kl[1]]
            atts['ksize'] = tf.AttrValue(list=tf.AttrValue.ListValue(i=ksl))
        if 'strides' in atts:
            st = atts['strides'].list.i
            stl = [st[0], st[2], st[3], st[1]]
            atts['strides'] = tf.AttrValue(list=tf.AttrValue.ListValue(i=stl))

        # Create new Operation
        op = sess.graph.create_op(op_type=n_org.type, inputs=op_inputs, name=n_org.name+'_new', attrs=atts) 
        out_tens = sess.graph.get_tensor_by_name(n_org.name+'_new'+':0')
        out_trans = tf.transpose(out_tens, [0, 3, 1, 2], name=n_org.name +'_transp_out')
        assert out_trans.shape == sess.graph.get_tensor_by_name(n_org.name+':0').shape
        
        # Update Connections
        out_nodes = [n for n in sess.graph.get_operations() if n_org.outputs[0] in n.inputs]
        for out in out_nodes:
            for j, nam in enumerate(out.inputs):
                if n_org.outputs[0] == nam:
                    out._update_input(j, out_trans)

    conv_names = [n.name for n in conv_nodes]
    conv_node_def = [n for n in graph_def.node if n.name in conv_names] 

    # Add conv_nodes to delete
    del_nodes.extend([n.node_def for n in conv_nodes])

    # Change T:Int64 Add AddV2 to T:Int32 
    add_nodes = [n for n in sess.graph.get_operations() if n.type in ['Add','AddV2'] and n.node_def.attr['T'].type == 9]
    for n_org in add_nodes:
        assert len(n_org.inputs)==2
        # Get Inputs
        org_inp_tens = [sess.graph.get_tensor_by_name(n_org.inputs[0].name), 
                        sess.graph.get_tensor_by_name(n_org.inputs[1].name)]

        to_int32 = lambda x: x if x.dtype != tf.int64 else tf.cast(x, dtype=tf.int32)
        inp_tens = list(map(to_int32, org_inp_tens))

        # Get Attributes
        atts = n_org.node_def.attr

        # Int32 Add operation
        atts['T'].type = types_pb2.DT_INT32
        
        # Create new Operation
        op = sess.graph.create_op(op_type=n_org.type, inputs=inp_tens, name=n_org.name+'_new', attrs=atts) 
        
        out_tens = sess.graph.get_tensor_by_name(n_org.name+'_new'+':0')
        assert out_tens.shape == sess.graph.get_tensor_by_name(n_org.name+':0').shape

        # Update Connections
        out_nodes = [n for n in sess.graph.get_operations() if n_org.outputs[0] in n.inputs]
        for out in out_nodes:
            for j, nam in enumerate(out.inputs):
                if n_org.outputs[0] == nam:
                    out._update_input(j, out_tens)

    # Add add_nodes to delete
    del_nodes.extend([n.node_def for n in add_nodes])

    # Change Tshape:Int64 Reshape to Tshape:Int32
    reshape_nodes = [n for n in sess.graph.get_operations() if n.type in ['Reshape'] and n.node_def.attr['Tshape'].type == 9]

    for n_org in reshape_nodes:
        assert len(n_org.inputs)==2
        # Get Inputs
        org_inp_tens = [sess.graph.get_tensor_by_name(n_org.inputs[0].name), 
                        sess.graph.get_tensor_by_name(n_org.inputs[1].name)]

        # to_int32 = lambda x: x if x.dtype != tf.int64 else tf.cast(x, dtype=tf.int32)
        # inp_tens = list(map(to_int32, org_inp_tens))

        # Get Attributes
        atts = n_org.node_def.attr
        # atts['T'].type does not accept DType
        # atts['Tshape'].type = tf.int32
        atts['Tshape'].type = types_pb2.DT_INT32

        # Create new Operation
        op = sess.graph.create_op(op_type=n_org.type, inputs=org_inp_tens, name=n_org.name+'_new', attrs=atts) 
        out_tens = sess.graph.get_tensor_by_name(n_org.name+'_new'+':0')
        assert out_tens.shape == sess.graph.get_tensor_by_name(n_org.name+':0').shape

        # Update Connections
        out_nodes = [n for n in sess.graph.get_operations() if n_org.outputs[0] in n.inputs]
        for out in out_nodes:
            for j, nam in enumerate(out.inputs):
                if n_org.outputs[0] == nam:
                    out._update_input(j, out_tens)

    # In case :0 in input names
    # by only names in graph_def, delete old nodes 
    reshape_names = [n.name for n in reshape_nodes]
    # reshape_nodes_after = [n for n in sess.graph.get_operations() if n.name in reshape_names] 
    graph_def = sess.graph.as_graph_def()
    reshape_node_def = [n for n in graph_def.node if n.name in reshape_names] 
    del_nodes.extend(reshape_node_def)

    # Delete nodes
    for on in del_nodes:
        graph_def.node.remove(on)

    # reshape_nodes = [n for n in graph_def.node if 'reshape' in n.name.lower()]
    
    # Write graph
    tf.io.write_graph(graph_def, "", graph_def_file.rsplit('.', 1)[0]+'_prep.pb', as_text=False)
