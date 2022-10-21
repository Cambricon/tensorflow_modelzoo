import numpy as np
import time
import os
import json
import copy
import csv
import tensorflow as tf
import glob


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


def calculate_topk_ratio(top1_sum, top5_sum, img_cnt):
    if img_cnt > 0:
        top1_ratio = top1_sum / img_cnt * 1.0
        top5_ratio = top5_sum / img_cnt * 1.0
        return (top1_ratio, top5_ratio)
    else:
        print("Error!denominator is 0!")
        exit(1)


def calculate_accuracy(data_dict, decode_res_list, img_label_map):
    # traverse the data_dict and the decode_res, to check
    # whether the pred_res == label
    # Notice: the data_dict may contain multi-batch data
    img_paths = data_dict["image_paths"]
    top1_cnt = 0
    top5_cnt = 0
    for img_path, decode_res in zip(img_paths, decode_res_list):
        img_label = get_img_label(img_path, img_label_map)
        if img_label != "None":
            decode_label_list = []
            for idx, item in enumerate(decode_res):
                decode_label_list.append(item[0])
            if img_label == decode_label_list[0]:
                top1_cnt += 1
                top5_cnt += 1
            elif img_label in decode_label_list:
                top5_cnt += 1
        else:
            pass
    return (top1_cnt, top5_cnt)


def accuracy(output, target, last_batch=0, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0) if last_batch == 0 else last_batch

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    if last_batch > 0:
        _, pred = output[0:last_batch].topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[0:last_batch].view(1, -1).expand_as(pred))

    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def save_result(
    imageNum,
    batch_size,
    top1,
    top5,
    meanAp,
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
        hardwareFps = imageNum / hardwaretime
        hwLatencyTime = hardwaretime / (imageNum / batch_size) * 1000
    if endToEndTime != TIME:
        e2eLatencyTime = endToEndTime / (imageNum / batch_size) * 1000
        endToEndFps = imageNum / endToEndTime
    result = {
        "Output": {
            "Accuracy": {
                "top1": "%.2f" % top1,
                "top5": "%.2f" % top5,
                "meanAp": "%.2f" % meanAp,
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
    tmp_res_2 = precision+"\t"+str(batch_size)+"\t"+ str(top1) + "/" + str(top5) + "\t" +str(hardwareFps)
    tmp_res = tmp_res_1 + tmp_res_2+"\n"
    if not os.path.exists(result_json):
        os.mknod(result_json)
    with open(result_json,"a") as f_obj:
        f_obj.write(tmp_res)

    #with open(result_json, "a") as outputfile:
    #    json.dump(result, outputfile, indent=4, sort_keys=True)
    #    outputfile.write("\n")
    #    outputfile.close()


def generate_img_label_map(img_label_file):
    img_label_map = {}
    with open(img_label_file, "r") as f_map:
        for line in f_map.readlines():
            line = line.strip("\n").split(" ")
            img_label_map[line[0]] = line[1]
    return img_label_map


def get_img_label(img_path, img_label_map):
    img_name = str(img_path).split("/")[-1]
    if img_name in img_label_map:
        return img_label_map[img_name]
    else:
        return "None"


def preprocess(img_file):
    img = tf.keras.preprocessing.image.load_img(img_file, target_size=[224, 224])
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = x[tf.newaxis, ...]
    x = tf.keras.applications.resnet.preprocess_input(x)
    return x


class Dataset:
    def __init__(self, image_dir, batch_size):
        if not os.path.isdir(image_dir):
            raise ValueError("image_dir:{} not exsist!".format(image_dir))
        self.image_paths = glob.glob(os.path.join(image_dir, "*/*.JPEG"))
        if len(self.image_paths) == 0:
            raise ValueError("image_dir does not contain any jpgc files")
        else:
            print("fetch {} nums images".format(len(self.image_paths)))
        self.batch_size = batch_size
        self.counter = 0

    def reset_conter(self):
        self.counter = 0

    def __decode(self, image_paths):
        tmp = []
        for image_path in image_paths:
            cur_img_tensor = preprocess(image_path)
            tmp.append(cur_img_tensor)
            if len(tmp) == 1:
                batch_tensor = cur_img_tensor
            else:
                batch_tensor = tf.concat([batch_tensor, cur_img_tensor], 0)

        return {"tensor": batch_tensor, "image_paths": image_paths}

    def get_batch(self):
        image_num = len(self.image_paths)
        if self.counter >= image_num:
            return None
        if self.counter + self.batch_size <= image_num:
            batch_image_paths = self.image_paths[
                self.counter : (self.counter + self.batch_size)
            ]
        else:
            batch_image_paths = self.image_paths[self.counter :]
            # batch_image_paths.extend(
            #    self.image_paths[0 : (self.counter + self.batch_size - image_num)]
            # )

        self.counter += self.batch_size
        return self.__decode(image_paths=batch_image_paths)


def get_res_savedmodel_dir(flags_obj):
    pub_converted_savedmodel_dir = flags_obj.model_dir
    # pub_converted_savedmodel_dir = args.converted_savedmodel_dir
    precision = flags_obj.quant_precision
    # precision = args.precision
    res_savedmodel_dir = pub_converted_savedmodel_dir + "/" + precision
    return res_savedmodel_dir
