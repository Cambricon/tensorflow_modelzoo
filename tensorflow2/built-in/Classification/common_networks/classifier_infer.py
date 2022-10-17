import sys
import os
import pprint
from tensorflow.python.compiler.magicmind import mm_convert as mm
from tensorflow.python.tools import saved_model_utils
import numpy as np
from absl import logging
import tensorflow as tf
import glob
import copy
import time

from absl import flags
from models.vision.vgg import common
from models.vision.preprocessing import preprocess_for_eval

cur_path = os.getcwd()
sys.path.append(cur_path + "/../../tools/")
import infer_flags
from saved_model_utils import *
from model_convert import *


sys.path.append(cur_path + "/../common/")
from infer_utils import *
from dataset.base import GetDataSetFromTFRecord
from dataset.imagenet import DATASET_DICT


def dataset_prepare(dataset_path):
    if not os.path.exists(dataset_path):
        sys.stderr.write(
            "Error:please prepare imagenet2012-val dataset in {}".format(dataset_path)
        )
        raise SystemExit(1)
    return 0

def get_dataset(data_dir, batch_size, data_ratio=1.0):
    datasetHandler = DATASET_DICT["imagenet"]
    datasetHandler.set_dataset_folder(data_dir)
    datasetHandler.set_batchsize(batch_size)
    datasetHandler.initialize()

    class DataPreprocess():
        def __init__(self):
            self.mean_subtract = True
            self.standardize = True

        def __call__(self,data):
            img_data = data.imageData
            return preprocess_for_eval(img_data,
                                       mean_subtract = self.mean_subtract,
                                       standardize = self.mean_subtract),data.image_label


    preprocess = DataPreprocess()
    return GetDataSetFromTFRecord(datasetHandler, preprocess, batch_size, data_ratio=data_ratio)


def get_native_savedmodel(native_savedmodel_dir):
    # check or get native savedmodel
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
    pb_file = flags_obj.pretrained_filepath + "/saved_model.pb"
    if os.path.exists(pb_file):
        model_convert(flags_obj, flags_obj.model_dir)
    return


def model_prepare(flags_obj):
    native_savedmodel_dir = flags_obj.pretrained_filepath
    get_native_savedmodel(native_savedmodel_dir)
    # use TFMM to infer when enable_eager=False
    if flags_obj.run_eagerly == False:
        get_converted_savedmodel(flags_obj)


def get_savedmodel_dir(flags_obj):
    if flags_obj.run_eagerly == False:
        savedmodel_dir = flags_obj.model_dir
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
    dataset = get_dataset(infer_data_dir, batch_size, flags_obj.data_ratio)
    results = {"image_paths": [], "predict_labels": []}

    print("..........inferencing..........")
    e2e_time = AverageMeter("E2e", ":6.5f")
    hardware_time = AverageMeter("Hardware", ":6.5f")
    total_e2e_time = 0
    total_hardware_time = 0
    top1_sum = 0
    top5_sum = 0
    # warmup
    for x, y in dataset:
        if warmup_count == 0:
            break
        ini_predict = graph_func(x)
        warmup_count -= 1

    iter_cnt = 0
    total_cnt = 0
    for x, y in dataset:
        e2e_time_begin = time.perf_counter()
        hardware_time_begin = time.perf_counter()
        ini_predict = graph_func(x)
        hardware_time_end = time.perf_counter()
        decode_res = decode_predictions(
            next(iter(ini_predict.values())).numpy(), top=top_k)
        e2e_time_end = time.perf_counter()

        iter_cnt += 1
        acc1, acc5, cur_img_cnt = calculate_accuracy(decode_res, y)
        total_cnt += cur_img_cnt
        top1_sum += acc1
        top5_sum += acc5
        cur_e2e_duration = e2e_time_end - e2e_time_begin
        cur_hardware_duration = hardware_time_end - hardware_time_begin
        e2e_time.update(cur_e2e_duration)
        hardware_time.update(cur_hardware_duration)
        total_e2e_time += cur_e2e_duration
        total_hardware_time += cur_hardware_duration


    avg_e2e_time = total_e2e_time / iter_cnt * 1.0
    avg_hw_time = total_hardware_time / iter_cnt * 1.0
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
    top1_ratio, top5_ratio = calculate_topk_ratio(top1_sum, top5_sum, total_cnt)
    save_result(
        total_cnt,
        batch_size,
        top1_ratio,
        top5_ratio,
        -1,
        total_hardware_time,
        total_e2e_time,
        result_json,
        net_name,
        precision,
        infer_mode,
    )
    print("..........inference results..........")
    print("%" * 85)
    print("top1_ratio is {},top5_ratio is {}".format(top1_ratio, top5_ratio))
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
