import numpy as np
import time
import os
import json
import copy
import sys
import csv
import tensorflow as tf
import glob
from collections import namedtuple
cur_path = os.getcwd()
sys.path.append(cur_path+ "/../transformer/models")
from official.nlp.data import squad_lib
sys.path.append(cur_path+ "/dataset")
from base import *
import squad_evaluate_v1_1
import squad_evaluate_v2_0
sys.path.append(cur_path+ "/models/model")
import tokenization

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
    f1,
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
                "f1": "%.2f" % f1,
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
    tmp_res_2 = precision+"\t"+str(batch_size)+"\t"+ str(f1) + "\t" +str(hardwareFps)
    tmp_res = tmp_res_1 + tmp_res_2+"\n"
    if not os.path.exists(result_json):
        os.mknod(result_json)
    with open(result_json,"a") as f_obj:
        f_obj.write(tmp_res)

class BertBasePreProcess:
    def __init__(self):
        pass
    def __call__(self,tensor):
        res={}
        res["input_word_ids"]=tensor["input_ids"]
        res["input_type_ids"]= tensor["segment_ids"]
        res["input_mask"] = tensor["input_mask"]
        return res,tensor["unique_ids"]

class SquadDataset(DatasetBase):
    def __init__(self):
        super(SquadDataset,self).__init__(None)
        self.max_seq_length=512
        self.max_query_length=64
        self.doc_stride = 128
        self.version_2_with_negative=False
        self.batchsize=None
        self.folder_path=None
        self.vocab_suffix="cased_L-12_H-768_A-12/vocab.txt"
        self.json_file_suffix="SQuAD/dev-v1.1.json"
        self.do_lower_case=False
        self.eval_examples=None
        self.eval_features=None
        self.tfrecord_gen_path="./tmp_record.tf_record"
        self.n_best_size=20
        self.max_answer_length=30

        self.feature_map={
            'unique_ids': tf.io.FixedLenFeature([],dtype=tf.int64),
            'input_ids': tf.io.FixedLenFeature([self.max_seq_length],dtype=tf.int64),
            'input_mask': tf.io.FixedLenFeature([self.max_seq_length],dtype=tf.int64),
            'segment_ids': tf.io.FixedLenFeature([self.max_seq_length],dtype=tf.int64)
        }

    def fetch_validation(self):
        return  self.tfrecord_gen_path

    def valid_dataset_path(self):
        if os.path.isdir(self.folder_path):
            return os.path.isfile(os.path.join(self.folder_path,self.json_file_suffix)) and \
                    os.path.isfile(os.path.join(self.folder_path,self.vocab_suffix))
        else:
            return False

    def set_dataset_folder(self,folder_path):
        self.folder_path=folder_path
        flag=self.valid_dataset_path()
        if not flag:
            raise ValueError("dataset folder:{} is not valid!".format(folder_path))

    def get_json_path(self):
        return os.path.join(self.folder_path,self.json_file_suffix)

    def initialize(self):
        tokenizer = tokenization.FullTokenizer(
                                    vocab_file=os.path.join(self.folder_path,self.vocab_suffix),
                                    do_lower_case=self.do_lower_case)
        self.eval_examples = squad_lib.read_squad_examples(
                        input_file=os.path.join(self.folder_path,self.json_file_suffix),
                        is_training=False,
                        version_2_with_negative=self.version_2_with_negative)

        eval_writer = squad_lib.FeatureWriter(
            filename= self.tfrecord_gen_path,
            is_training=False)
        self.eval_features = []

        def _append_feature(feature, is_padding):
            if not is_padding:
                self.eval_features.append(feature)
            eval_writer.process_feature(feature)

        kwargs = dict(
            examples=self.eval_examples,
            tokenizer=tokenizer,
            max_seq_length=self.max_seq_length,
            doc_stride=self.doc_stride,
            max_query_length=self.max_query_length,
            is_training=False,
            output_fn=_append_feature,
            batch_size=self.batchsize)

        _ = squad_lib.convert_examples_to_features(**kwargs)
        eval_writer.close()


    def set_batchsize(self,batchsize):
        self.batchsize=batchsize

    def decode(self,record)-> RawData:
        obj = tf.io.parse_single_example(record, self.feature_map)
        res={
            "input_mask":tf.cast(obj["input_mask"],tf.int32),
            "input_ids":tf.cast(obj["input_ids"],tf.int32),
            "unique_ids": tf.cast(obj["unique_ids"],tf.int32),
            "segment_ids": tf.cast(obj["segment_ids"],tf.int32)
        }
        return res

class TFDatasetGenHelper:
    def __init__(self,dataset,preprocess):
        self.dataset=dataset
        self.preprocess=preprocess
    def __call__(self,tfrecord):
        return self.preprocess(self.dataset.decode(tfrecord))

def get_dataset_from_tfrecord(dataset,preprocess,batch_size,parallel=32):
    files=dataset.fetch_validation()
    tf_dataset_helper=TFDatasetGenHelper(dataset,preprocess)
    dataset=tf.data.TFRecordDataset(files)
    dataset=dataset.map(map_func=tf_dataset_helper, num_parallel_calls=parallel)
    return dataset.batch(batch_size=batch_size,drop_remainder=True)

def get_dataset(flags_obj):
    ds = SquadDataset()
    ds.set_dataset_folder(flags_obj.data_dir)
    ds.set_batchsize(flags_obj.batch_size)
    ds.initialize()
    preprocess = BertBasePreProcess()
    return get_dataset_from_tfrecord(ds, preprocess, flags_obj.batch_size)


BertBaseRawResult = namedtuple('BertBaseRawResult',
                                   ['unique_id', 'start_logits', 'end_logits'])

class BertBasePostProcess:
    def __init__(self):
        self.output_tensor_map={
            "end_positions":"output_0",
            "start_positions":"output_1"
        }
    def __get_tensor(self,tensor_key,infer_res):
        assert tensor_key in self.output_tensor_map.keys()
        if tensor_key in infer_res.keys():
            return infer_res[tensor_key]
        elif self.output_tensor_map[tensor_key] in infer_res.keys():
            return infer_res[self.output_tensor_map[tensor_key]]
        else:
            raise ValueError("cannot find:[{}] in infer res, infer_res keys:{} postprocess tensor map:{}"
                            .format(tensor_key,infer_res.keys(),self.output_tensor_map)
                            )

    def __call__(self,infer_res):
        res=[]
        start_positions=self.__get_tensor("start_positions",infer_res)
        end_positions=self.__get_tensor("end_positions",infer_res)
        unique_ids=infer_res["unique_ids"]
        for values in zip(unique_ids.numpy(), start_positions.numpy(),
                    end_positions.numpy()):
            res.append(BertBaseRawResult(unique_id=values[0],
                                        start_logits=values[1].tolist(),
                                        end_logits=values[2].tolist()))
        return res

def calc_hits(flags_obj, total_results):
    ds = SquadDataset()
    ds.set_dataset_folder(flags_obj.data_dir)
    ds.set_batchsize(flags_obj.batch_size)
    ds.initialize()

    eval_json_path=os.path.join(flags_obj.data_dir, "SQuAD/dev-v1.1.json")
    with open(eval_json_path,"r") as f:
        pred_dataset=json.load(f)["data"]

    all_predictions, all_nbest_json, scores_diff_json=squad_lib.postprocess_output(
                                ds.eval_examples,
                                ds.eval_features,
                                total_results,
                                ds.n_best_size,
                                ds.max_answer_length,
                                ds.do_lower_case,
                                version_2_with_negative=ds.version_2_with_negative,
                                null_score_diff_threshold=0.0,
                                verbose= False)
    if ds.version_2_with_negative:
        return squad_evaluate_v2_0.evaluate(pred_dataset, all_predictions,
                                            scores_diff_json)
    else:
        return squad_evaluate_v1_1.evaluate(pred_dataset, all_predictions)["final_f1"]*100.0

def get_res_savedmodel_dir(flags_obj):
    pub_converted_savedmodel_dir = flags_obj.model_dir
    precision = flags_obj.quant_precision
    res_savedmodel_dir = pub_converted_savedmodel_dir + "/" + precision
    return res_savedmodel_dir
