#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-05-20 15:35:27
#   Description : coco评测
#
# ================================================================
import os
import json
import sys
import cv2
import shutil
import logging
logger = logging.getLogger(__name__)


clsid2catid = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16,
               15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31,
               27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43,
               39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56,
               51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72,
               63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85,
               75: 86, 76: 87, 77: 88, 78: 89, 79: 90}

catid2clsid = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14,
               17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26,
               32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38,
               44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50,
               57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62,
               73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74,
               86: 75, 87: 76, 88: 77, 89: 78, 90: 79}

def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def cocoapi_eval(jsonfile,
                 style,
                 coco_gt=None,
                 anno_file=None,
                 max_dets=(100, 300, 1000)):
    """
    Args:
        jsonfile: Evaluation json file, eg: bbox.json, mask.json.
        style: COCOeval style, can be `bbox` , `segm` and `proposal`.
        coco_gt: Whether to load COCOAPI through anno_file,
                 eg: coco_gt = COCO(anno_file)
        anno_file: COCO annotations file.
        max_dets: COCO evaluation maxDets.
    """
    assert coco_gt != None or anno_file != None
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if coco_gt == None:
        coco_gt = COCO(anno_file)
    logger.info("Start evaluate...")
    coco_dt = coco_gt.loadRes(jsonfile)
    if style == 'proposal':
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.useCats = 0
        coco_eval.params.maxDets = list(max_dets)
    else:
        coco_eval = COCOeval(coco_gt, coco_dt, style)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats

def coco_eval(data, anno_file="", is_pkl=False):
    import numpy as np
    if is_pkl:
        import pickle as pkl
        with open(data, "rb") as f:
            pkl_data = pkl.load(f)
    else:
        pkl_data = data
    source_id = np.concatenate([pkl_data[i][0] for i in range(len(pkl_data))], axis=0)
    boxes = np.concatenate([pkl_data[i][1] for i in range(len(pkl_data))], axis=0)
    scores = np.concatenate([pkl_data[i][2] for i in range(len(pkl_data))], axis=0)
    classes = np.concatenate([pkl_data[i][3] for i in range(len(pkl_data))], axis=0)
    nums = np.concatenate([pkl_data[i][4] for i in range(len(pkl_data))], axis=0)
    scale = np.concatenate([pkl_data[i][5] for i in range(len(pkl_data))], axis=0)
    result = []
    print(classes.max(), "============")
    print(classes.min(), "============")
    for i in range(len(source_id)):
        for j in range(nums[i]):
            ymin, xmin, ymax, xmax = boxes[i][j]
            h_scale, w_scale = scale[:, 2, :][i]
            clsid = classes[i][j]
            score = scores[i][j]
            catid = int(clsid)
            w = xmax - xmin + 1
            h = ymax - ymin + 1

            bbox = [xmin / w_scale, ymin / h_scale, w / w_scale, h / h_scale]
            # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
            bbox = [round(float(x) * 10) / 10 for x in bbox]
            result.append([int(source_id[i]), *bbox, score, catid])
    ann_numpy = np.array(result)
    print(ann_numpy.shape)

    cocoapi_eval(ann_numpy, "bbox", anno_file=anno_file)



if __name__ == "__main__":
    coco_eval("eval.pkl", is_pkl=True)
