import tensorflow as tf
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import tensor_shape
from tensorflow.python.saved_model import save
import logging

SIGNITURE_FUNC = 'serving_default'
TMP_FOLDER="./tmp"

def get_tag_sets(saved_model_dir):
    return saved_model_utils.get_saved_model_tag_sets(saved_model_dir)

def has_dynamic_dim(shape_list):
    for shape in shape_list:
        if isinstance(shape,tensor_shape.TensorShape):
            shape=list(shape)
        for dim in shape:
            if dim is None or dim == -1:
                return True
    return False

def get_signature_keys(saved_model_dir:str,tag_set=None):
    res=[]
    saved_model_pb=saved_model_utils.read_saved_model(saved_model_dir)
    if len(saved_model_pb.meta_graphs)==0:
        raise ValueError("saved_model_dir:{} dose not contains any meta graph".format(saved_model_dir))
    meta_graph=None
    if len(saved_model_pb.meta_graphs)==1:
        meta_graph=saved_model_pb.meta_graphs[0]
    else:
        if tag_set is None:
            raise RuntimeError("saved_model_dir:{} contains more than one meta graph, need use tag_set to filter")
        set_of_tags = set([tag for tag in tag_set.split(",") if tag])
        for meta_graph_def in saved_model_pb.meta_graphs:
            if set(meta_graph_def.meta_info_def.tags) == set_of_tags:
                meta_graph=meta_graph_def
                break
        if meta_graph is None:
            raise RuntimeError("MetaGraphDef associated with tag-set %r could not be"
                            " found in SavedModel" % tag_set)
    for key in meta_graph.signature_def.keys():
        res.append(key)
    return res


def get_model_inputoutput_shape(saved_model_dir:str,tag_set=None,signature=SIGNITURE_FUNC):

    res={'inputs':{},'outputs':{}}
    saved_model_pb=saved_model_utils.read_saved_model(saved_model_dir)
    if len(saved_model_pb.meta_graphs)==0:
        raise ValueError("saved_model_dir:{} dose not contains any meta graph".format(saved_model_dir))
    signature_def=None
    if len(saved_model_pb.meta_graphs)==1:
        signature_def=saved_model_pb.meta_graphs[0].signature_def[signature]
    else:
        if tag_set is None:
            raise RuntimeError("saved_model_dir:{} contains more than one meta graph, need use tag_set to filter")
        set_of_tags = set([tag for tag in tag_set.split(",") if tag])
        for meta_graph_def in saved_model_pb.meta_graphs:
            if set(meta_graph_def.meta_info_def.tags) == set_of_tags:
                signature_def=meta_graph_def.signature_def[signature]
                break
    if signature_def is None:
        raise RuntimeError("MetaGraphDef associated with tag-set %r could not be"
                        " found in SavedModel" % tag_set)
    inputs_tensor_info=signature_def.inputs
    outputs_tensor_info=signature_def.outputs
    for key, item in inputs_tensor_info.items():
        res['inputs'][key]=[dim.size for dim in item.tensor_shape.dim]
    for key, item in outputs_tensor_info.items():
        res['outputs'][key]=[dim.size for dim in item.tensor_shape.dim]
    return res

def savedmodel_dynamicbatch_to_fixbatch(src_dir,out_dir,batchsize,signature=SIGNITURE_FUNC,tag=["serve"]):
    loadModel=tf.saved_model.load(src_dir,tags=tag)
    graph_func = loadModel.signatures[signature]
    graph_func=convert_to_constants.convert_variables_to_constants_v2(graph_func)
    #changing tensor shapes
    for idx in range(len(graph_func.inputs)):
        tensor=graph_func.inputs[idx]
        shape=list(tensor.shape)
        if shape[0] is None:
            shape[0]=batchsize
        graph_func.inputs[idx].set_shape(tensor_shape.TensorShape(shape))

    for idx in range(len(graph_func.outputs)):
        tensor=graph_func.outputs[idx]
        shape=list(tensor.shape)
        if shape[0] is None:
            shape[0]=batchsize
        graph_func.outputs[idx].set_shape(tensor_shape.TensorShape(shape))

    #saving a new saved_model
    signatures={key: value for key, value in loadModel.signatures.items()}
    signatures[signature]=graph_func
    save.save(loadModel,out_dir,signatures)
    return out_dir

def savedmodel_fix_dynamicmodel(src_dir,out_dir,input_shapes:dict,output_shapes=None,signature=SIGNITURE_FUNC,tag=["serve"]):
    loadModel=tf.saved_model.load(src_dir,tags=tag)
    graph_func = loadModel.signatures[signature]
    graph_func=convert_to_constants.convert_variables_to_constants_v2(graph_func)
    #changing tensor shapes
    for idx in range(len(graph_func.inputs)):
        tensor=graph_func.inputs[idx]
        name=str(tensor.name).split(":",1)[0]
        if name in input_shapes.keys():
            graph_func.inputs[idx].set_shape(tensor_shape.TensorShape(input_shapes[name]))
        else:
            logging.error("cannot get shape in input shapes for input tensor:{}".format(name))

    if output_shapes is not None and isinstance(output_shapes,dict):
        for idx in range(len(graph_func.outputs)):
            tensor=graph_func.outputs[idx]
            name=str(tensor.name).split(":",1)[0]
            if name in output_shapes.keys():
                graph_func.outputs[idx].set_shape(tensor_shape.TensorShape(output_shapes[name]))
            else:
                logging.error("cannot get shape in output shapes for output tensor:{}".format(name))

    #saving a new saved_model
    signatures={key: value for key, value in loadModel.signatures.items()}
    signatures[signature]=graph_func
    save.save(loadModel,out_dir,signatures)
    return out_dir

def savedmodel_derive_middleLayer_output(src_dir,
                        out_dir,
                        filter_node_fn=None,
                        filter_tensor_fn=None,
                        middle_layer_list=[],
                        tag=["serve"],
                        signature_key=SIGNITURE_FUNC,
                        middle_out_preffix="|diff|"):
    loadModel=tf.saved_model.load(src_dir,tags=tag)
    graph_func = loadModel.signatures[signature_key]
    graph_func=convert_to_constants.convert_variables_to_constants_v2(graph_func)
    inner_input=graph_func.inputs
    inner_output=graph_func.outputs

    inputs={}
    outputs={}

    for key in inner_input:
        inputs[key.name.split(":")[0]]=key.name
    for key in inner_output:
        outputs[key.name.split(":")[0]]=key.name
    input_node={}
    output_node={}

    graph=graph_func.graph
    with tf.compat.v1.Session(graph=graph) as sess:

        inner_node_name=[]
        if filter_node_fn is not None:
            total_node_names=[n.name for n in graph.as_graph_def().node if filter_node_fn(n)]
        else:
            total_node_names=[n.name for n in graph.as_graph_def().node]
        for node in middle_layer_list:
            for inner_node in total_node_names:
                if node in inner_node:
                    inner_node_name.append(inner_node)

        logging.info("found :{} matching {}".format(inner_node_name,middle_layer_list))
        if len(inner_node_name) == 0:
            raise ValueError("cannot found node matching middle layers:{}".format(middle_layer_list))
        for key in inputs.keys():
            input_node[key]=graph.get_tensor_by_name(inputs[key])

        for idx,node_name in enumerate(inner_node_name):
            tensor = graph.get_tensor_by_name(node_name+":0")
            if filter_tensor_fn is not None:
                if filter_tensor_fn(tensor):
                    output_node[middle_out_preffix+"_{}".format(idx)]=tensor
                else:
                    continue
            else:
                output_node[middle_out_preffix+"_{}".format(idx)]=tensor
        for key in outputs.keys():
            output_node[key]=graph.get_tensor_by_name(outputs[key])
        tf.compat.v1.saved_model.simple_save(sess, out_dir, input_node, output_node)
    return out_dir

if __name__== "__main__":
    pass
