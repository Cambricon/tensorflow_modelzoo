import numpy as np
import time
import os
import json
import copy
import sys
import csv
import tensorflow as tf
import glob
cur_path = os.getcwd()
sys.path.append(cur_path + "/../")
from dataset.wmt14_newstest2014 import *

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def save_result(
    input_count,
    batch_size,
    gleu,
    hardwaretime,
    endToEndTime,
    result_json,
    net_name,
    precision,
    mode,
):

    TIME = -1
    hardwareFps = -1
    hwLatencyTime = -1
    endToEndFps = -1
    e2eLatencyTime = -1
    if hardwaretime != TIME:
        hardwareFps = input_count / hardwaretime
        hwLatencyTime = hardwaretime / (input_count / batch_size) * 1000
    if endToEndTime != TIME:
        e2eLatencyTime = endToEndTime / (input_count / batch_size) * 1000
        endToEndFps = input_count / endToEndTime
    result = {
        "Output": {
            "Accuracy": {
                "GLEU": "%.2f" % gleu,
            },
            "HostLatency(ms)": {
                "average": "%.2f" % e2eLatencyTime,
                "throughput(fps)": "%.2f" % endToEndFps,
            },
            "HardwareLatency(ms)": {
                "average": "%.2f" % hwLatencyTime,
                "throughput(fps)": "%.2f" % hardwareFps,
            },
            "BasicInfo": {
                "net_name": "%s" % net_name,
                "mode": "%s" % mode,
                "batch_size": "%d" % batch_size,
                "quant_precision": "%s" % precision,
            },
        }
    }

    tmp_res_1 = net_name +"\t"+ mode +"\t"
    tmp_res_2 = precision+"\t"+str(batch_size)+"\t"+ str(gleu) + "\t" +str(hardwareFps)
    tmp_res = tmp_res_1 + tmp_res_2+"\n"
    if not os.path.exists(result_json):
        os.mknod(result_json)
    with open(result_json,"a") as f_obj:
        f_obj.write(tmp_res)


class WrapNewstest2014(object):
    """ """
    def __init__(self, batch_size, total_batch_num =None, data_root_dir = None):
        self.batch_size = batch_size
        self.total_batch_num = total_batch_num
        self.iter = self.get_iter(batch_size, total_batch_num)
        self.data_root_dir = data_root_dir

    def __iter__(self):
        """ """
        return self.iter

    def ResetCounter(self):
        """ """
        self.iter = self.get_iter(self.batch_size, self.total_batch_num)

    def get_iter(self, batch_size, total_batch_num):
        """ """
        json_dict["batch_size"] = batch_size
        ds = Newstest2014Encode(json_dict, None,
            {'transformer_parameters': 'None',
            'max_seq_length': 128,
            'encode': 'newstest2014.en',
            'decode': 'newstest2014.de',
            'vocab': 'vocab.ende.32768'},
            data_root_dir = self.data_root_dir)
        loader = ds.load_query_samples()
        data_dict ={"input_tensor":None}
        i = 0
        for x, y in loader:
            data_dict["input_tensor"] = tf.constant(
                        np.array(x).astype(np.int32))
            if None!=total_batch_num and i == total_batch_num:
                raise StopIteration
            i+=1
            yield (data_dict, y)

def GetDataSet(flags_obj):
    return WrapNewstest2014(flags_obj.batch_size, data_root_dir = flags_obj.data_dir, total_batch_num=None)

def get_res_savedmodel_dir(flags_obj):
    pub_converted_savedmodel_dir = flags_obj.model_dir
    precision = flags_obj.quant_precision
    res_savedmodel_dir = pub_converted_savedmodel_dir + "/" + precision
    return res_savedmodel_dir
