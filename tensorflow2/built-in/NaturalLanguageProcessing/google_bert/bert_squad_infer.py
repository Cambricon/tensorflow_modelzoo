import sys
import os
import pprint
from tensorflow.python.compiler.magicmind import mm_convert as mm
from tensorflow.python.tools import saved_model_utils
import numpy as np
from absl import logging
import glob
import copy
import time
from dataset.squad import *
from utils.infer_utils import *
import tensorflow as tf

from absl import flags
cur_path = os.getcwd()
sys.path.append(cur_path + "/../../tools/")
from saved_model_utils import *
from model_convert import *
import infer_flags

input_infos={
    "input_mask":{
        "shape":[-1,512],   #[batchsize,max_seq_length]
        "dtype":np.int32,
        "range":[0,1.9]
    },
    "input_type_ids":{
         "shape":[-1,512],  #[batchsize,max_seq_length]
         "dtype":np.int32,
         "range":[0,1.9]
    },
    "input_word_ids":{
         "shape":[-1,512],  #[batchsize,max_seq_length]
         "dtype":np.int32,
         "range":[0,28306]
    },
}

def dataset_prepare(dataset_path):
    if not os.path.exists(dataset_path):
        sys.stderr.write(
            "Error:please prepare newstest2014 dataset in {}".format(dataset_path)
        )
        raise SystemExit(1)
    return 0


def get_native_savedmodel(native_savedmodel_dir):
    # check savedmodel
    native_savedmodel_dir = flags_obj.pretrained_filepath
    dest_file = native_savedmodel_dir + "/saved_model.pb"
    if not os.path.exists(dest_file):
        logging.info("cambricon-note: Please prepare saved_model.")
    else:
        logging.info(
            "cambricon-note: keras saved_model already exists,now start to convert to TFMM model!"
        )
    return


def get_converted_savedmodel(flags_obj):
    # generate savedmodel and save to disk.
    res_savedmodel_dir = get_res_savedmodel_dir(flags_obj)
    pb_file = res_savedmodel_dir + "/saved_model.pb"
    if not os.path.exists(pb_file):
        model_convert(flags_obj, res_savedmodel_dir, input_infos)
    return


def model_prepare(flags_obj):
    native_savedmodel_dir = flags_obj.pretrained_filepath
    get_native_savedmodel(native_savedmodel_dir)
    # use TFMM to infer when run_eagerly=False
    if flags_obj.run_eagerly == False:
        get_converted_savedmodel(flags_obj)


def get_savedmodel_dir(flags_obj):
    if flags_obj.run_eagerly == False:
        savedmodel_dir = get_res_savedmodel_dir(flags_obj)
    else:
        savedmodel_dir = flags_obj.pretrained_filepath
    return savedmodel_dir


def model_infer(flags_obj, signature="serving_default"):
    print("..........loading converted model..........")
    if flags_obj.quant_precision == "fp16":
        from tensorflow.python.framework import config
        config.set_optimizer_experimental_options({'auto_mixed_precision': True})

    savedmodel_dir = get_savedmodel_dir(flags_obj)
    infer_data_dir = flags_obj.data_dir
    batch_size = int(flags_obj.batch_size)
    top_k = flags_obj.top_k
    warmup_count = flags_obj.warmup_count

    model = tf.saved_model.load(savedmodel_dir, tags=mm.tag_constants.SERVING)
    graph_func = model.signatures["serving_default"]
    dataset = get_dataset(flags_obj)

    print("..........inferencing..........")
    e2e_time = AverageMeter("E2e", ":6.5f")
    hardware_time = AverageMeter("Hardware", ":6.5f")
    total_e2e_time = 0
    total_hardware_time = 0

    total_results = []
    input_count = 0
    postprocess = BertBasePostProcess()
    for x, y in dataset:
        e2e_time_begin = time.perf_counter()
        hardware_time_begin = time.perf_counter()
        predicts = graph_func(**x)
        predicts["unique_ids"] = y
        batch_results = postprocess(predicts)
        total_results.extend(batch_results)
        hardware_time_end = time.perf_counter()
        e2e_time_end = time.perf_counter()

        input_count += 1
        cur_e2e_duration = e2e_time_end - e2e_time_begin
        cur_hardware_duration = hardware_time_end - hardware_time_begin
        e2e_time.update(cur_e2e_duration)
        hardware_time.update(cur_hardware_duration)
        total_e2e_time += cur_e2e_duration
        total_hardware_time += cur_hardware_duration

    f1 = calc_hits(flags_obj, total_results)
    avg_e2e_time = total_e2e_time / input_count * 1.0
    avg_hw_time = total_hardware_time / input_count * 1.0
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
    print("avg_e2e_time is:{},class avg e2e time:{}".format(avg_e2e_time, e2e_time.avg))

    result_json = flags_obj.result_path
    net_name = flags_obj.model
    infer_mode = "jit" if flags_obj.run_eagerly == False else "eager"
    precision = flags_obj.quant_precision
    save_result(
        input_count,
        batch_size,
        f1,
        total_hardware_time,
        total_e2e_time,
        result_json,
        net_name,
        precision,
        infer_mode,
    )
    print("..........inference results..........")
    print("%" * 85)
    print("f1 is {}".format(f1))
    print("%" * 85)


# --- main ---

flags_obj = flags.FLAGS
flags_obj(sys.argv)
# dataset prepare
dataset_prepare(flags_obj.data_dir)
# model prepare
model_prepare(flags_obj)
# infererence process
model_infer(flags_obj)
