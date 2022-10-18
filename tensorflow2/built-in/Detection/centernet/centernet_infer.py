import sys
import os
import pprint
from tensorflow.python.compiler.magicmind import mm_convert as mm
from tensorflow.python.tools import saved_model_utils
import numpy as np
import argparse
import tensorflow as tf
from absl import logging
import glob
import copy
import time
from utils.infer_utils import *
from absl import flags

cur_path = os.getcwd()
models_path = cur_path + "/models/"
sys.path.append(models_path)
import object_detection

sys.path.append(cur_path + "/../../tools/")
import infer_flags
from model_convert import *
from saved_model_utils import *


def get_centernet_config(data_dir, batch_size=1):
    from object_detection import model_lib
    MODEL_BUILD_UTIL_MAP = model_lib.MODEL_BUILD_UTIL_MAP
    pipeline_config_path = os.environ["PIPELINE_CONFIG_PATH"]
    label_map_path = os.environ["LABEL_MAP_PATH"]
    eval_tfrecords_path = data_dir
    assert os.path.isfile(pipeline_config_path),\
        "The pipe line config file: {} does not exist.".format(
            pipeline_config_path)
    get_configs_from_pipeline_file = MODEL_BUILD_UTIL_MAP['get_configs_from_pipeline_file']
    try:
        configs = get_configs_from_pipeline_file(pipeline_config_path)
    except:
        raise ValueError("Failed to parse pipeline config specified in {}".format(
            pipeline_config_path))

    eval_index = 0
    if hasattr(configs['eval_input_configs'][eval_index], "label_map_path"):
        if not os.path.isfile(configs['eval_input_configs'][eval_index].label_map_path):
            configs['eval_input_configs'][eval_index].label_map_path = label_map_path
    else:
        raise ValueError("eval_input_config has no attribute label_map_path, please check"
                         " the configuration file mscoco_label_map.pbtxt")
    if hasattr(configs['eval_input_configs'][eval_index], "tf_record_input_reader"):
        if not os.path.isfile(configs['eval_input_configs'][eval_index].tf_record_input_reader.input_path[0]):
            configs['eval_input_configs'][eval_index].tf_record_input_reader.ClearField(
                "input_path")
            configs['eval_input_configs'][eval_index].tf_record_input_reader.input_path.append(
                eval_tfrecords_path)
    else:
        raise ValueError("eval_input_config has no attribute tf_record_input_reader, "
                         "please check the configuration file mscoco_label_map.pbtxt")

    if batch_size != 1:
        if hasattr(configs['eval_config'], "batch_size"):
            configs['eval_config'].batch_size = batch_size

    return configs


def get_centernet_coco17_dataLoader(configs):
    assert len(configs) > 0, "invalid configs."
    from object_detection import inputs
    from object_detection import model_lib
    MODEL_BUILD_UTIL_MAP = model_lib.MODEL_BUILD_UTIL_MAP
    model_config = configs['model']
    eval_config = configs['eval_config']
    eval_input_configs = configs['eval_input_configs']
    eval_index = 0
    eval_input_config = eval_input_configs[eval_index]
    detection_model = MODEL_BUILD_UTIL_MAP['detection_model_fn_base'](
        model_config=model_config, is_training=False)
    try:
        eval_input = inputs.eval_input(
            eval_config=eval_config,
            eval_input_config=eval_input_config,
            model_config=model_config,
            model=detection_model)
    except:
        raise ValueError("Failed to generate coco17 dataloader")
    return eval_input


def get_native_savedmodel(native_savedmodel_dir):
    # check savedmodel
    dest_file = native_savedmodel_dir + "/saved_model.pb"
    if not os.path.exists(dest_file):
        logging.info("cambricon-note: Please prepare saved_model.")
    else:
        logging.info(
            "cambricon-note: keras saved_model already exists, now start to convert to TFMM model!"
        )
    return


def get_converted_savedmodel(flags_obj):
    pb_file = flags_obj.pretrained_filepath + "/saved_model.pb"
    if os.path.exists(pb_file):
        model_convert(flags_obj, flags_obj.model_dir)
    return


def get_post_process_function(args, configs):
    OUT_STRIDE = 4
    if args.run_eagerly == True:
        output_map = {"object_center": "object_center",
                      "box_scale": "box/scale",
                      "box_offset": "box/offset"}
    else:
        output_map = {"object_center": "output_2",
                      "box_scale": "output_1",
                      "box_offset": "output_0"}

    max_box_predictions = \
        configs['model'].center_net.object_center_params.max_box_predictions
    return CenternetPostProcess(args, OUT_STRIDE, max_box_predictions, output_map=output_map)


def model_prepare(args):
    native_savedmodel_dir = args.pretrained_filepath
    get_native_savedmodel(native_savedmodel_dir)
    # use TFMM to infer when enable_eager=False
    if args.run_eagerly == False:
        get_converted_savedmodel(args)


def get_savedmodel_dir(flags_obj):
    if flags_obj.run_eagerly == False:
        savedmodel_dir = flags_obj.model_dir
    else:
        savedmodel_dir = flags_obj.pretrained_filepath
    return savedmodel_dir


def model_infer(args, signature="serving_default"):
    print("..........loading converted model..........")
    if flags_obj.quant_precision == "fp16":
        from tensorflow.python.framework import config
        config.set_optimizer_experimental_options({'auto_mixed_precision': True})

    savedmodel_dir = get_savedmodel_dir(args)

    infer_data_dir = args.data_dir
    batch_size = int(args.batch_size)
    configs = get_centernet_config(infer_data_dir, batch_size)
    data_loader = get_centernet_coco17_dataLoader(configs)
    post_process_function = get_post_process_function(args, configs)
    evaluators = Evaluators(args, configs)
    warmup_count = args.warmup_count

    model = tf.saved_model.load(savedmodel_dir, tags=mm.tag_constants.SERVING)
    graph_func = model.signatures["serving_default"]

    print("..........inferencing..........")
    e2e_time = AverageMeter("E2e", ":6.5f")
    hardware_time = AverageMeter("Hardware", ":6.5f")
    total_e2e_time = 0
    total_hardware_time = 0
    # warmup
    i = 0
    for features, labels in data_loader:
        if i > warmup_count:
            break
        i += 1
        ini_predict = graph_func(features["image"])

    cnt = 0
    for features, labels in data_loader:
        e2e_time_begin = time.perf_counter()
        hardware_time_begin = time.perf_counter()
        prediction_dict = graph_func(features["image"])
        hardware_time_end = time.perf_counter()
        eval_dict = post_process_function(prediction_dict, features, labels)
        e2e_time_end = time.perf_counter()
        cur_e2e_duration = e2e_time_end - e2e_time_begin
        cur_hardware_duration = hardware_time_end - hardware_time_begin
        e2e_time.update(cur_e2e_duration)
        hardware_time.update(cur_hardware_duration)
        total_e2e_time += cur_e2e_duration
        total_hardware_time += cur_hardware_duration
        cnt += 1
        evaluators.FeedEvalData2Evaluators(eval_dict)

    acc = evaluators.DoEvaluation()
    avg_e2e_time = total_e2e_time / cnt * 1.0
    avg_hw_time = total_hardware_time / cnt * 1.0
    print(
        "total_hardware_time is:{},class total hw time:{}".format(
            total_hardware_time, hardware_time.sum
        )
    )
    print(
        "total_e2e_time is:{},class total e2e time:{}".format(
            total_e2e_time, e2e_time.sum
        )
    )
    print(
        "avg_hardware_time is:{},class avg hw time:{}".format(
            avg_hw_time, hardware_time.avg
        )
    )
    print("avg_e2e_time is:{},class avg e2e time:{}".format(
        avg_e2e_time, e2e_time.avg))

    result_json = args.result_path
    net_name = args.model
    infer_mode = "jit" if args.run_eagerly == False else "eager"
    precision = args.quant_precision
    save_result(
        cnt,
        batch_size,
        acc,
        total_hardware_time,
        total_e2e_time,
        result_json,
        net_name,
        precision,
        infer_mode,
    )
    print("..........inference results..........")
    print("%" * 85)
    print(acc)
    print("%" * 85)


# --- main ---
flags_obj = flags.FLAGS
flags_obj(sys.argv)
# model prepare
model_prepare(flags_obj)
# infererence process
model_infer(flags_obj)
