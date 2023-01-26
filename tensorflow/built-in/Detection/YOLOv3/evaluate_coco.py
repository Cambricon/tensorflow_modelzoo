#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : evaluate.py
#   Author      : YunYang1994
#   Created date: 2019-02-21 15:30:26
#   Description :
#
#================================================================

import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import models.core.utils as utils
from models.core.config import cfg
from models.core.yolov3 import YOLOV3
from pycocotools.cocoeval import COCOeval

flags = tf.flags
FLAGS = None

class YoloTest(object):
    def __init__(self):
        self.input_size       = cfg.TEST.INPUT_SIZE
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes          = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes      = len(self.classes)
        self.anchors          = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.score_threshold  = cfg.TEST.SCORE_THRESHOLD
        self.iou_threshold    = cfg.TEST.IOU_THRESHOLD
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.annotation_path  = cfg.TEST.ANNOT_PATH
        self.weight_file      = cfg.TEST.WEIGHT_FILE
        self.write_image      = cfg.TEST.WRITE_IMAGE
        self.write_image_path = cfg.TEST.WRITE_IMAGE_PATH
        self.show_label       = cfg.TEST.SHOW_LABEL
        self.instances_path   = cfg.TEST.INSTANCES_PATH

        with tf.name_scope('input'):
            self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.trainable  = tf.placeholder(dtype=tf.bool,    name='trainable')

        with tf.name_scope("define_loss"):
          model = YOLOV3(self.input_data, self.trainable)
        self.pred_sbbox, self.pred_mbbox, self.pred_lbbox = model.pred_sbbox, model.pred_mbbox, model.pred_lbbox
        """
        with tf.name_scope('ema'):
            ema_obj = tf.train.ExponentialMovingAverage(self.moving_ave_decay)
        """

        config = tf.ConfigProto()

        self.sess = tf.Session(config=config)

        self.saver = tf.train.Saver() # ema_obj.variables_to_restore())
        self.saver.restore(self.sess, self.weight_file)

    def predict(self, image):

        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox],
            feed_dict={
                self.input_data: image_data,
                self.trainable: False
            }
        )

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)

        return bboxes
    def evaluate(self):
        result = []
        from pycocotools.coco import COCO
        coco = COCO(self.instances_path)
        cats = coco.loadCats(coco.getCatIds())
        classname2id = {}
        for cat in cats:
            classname2id[cat['name']] = cat['id']

        imgname2id = {}
        imgIds = coco.getImgIds()
        imgs = coco.loadImgs(imgIds)
        imgname2id = {}
        for img in imgs:
            imgname = img['file_name']
            imgid = img['id']
            imgname2id[imgname] = imgid
        imgIds = []
        with open(self.annotation_path, 'r') as annotation_file:
            for num, line in enumerate(annotation_file):
                if num % 100 == 0:
                    print(num)
                annotation = line.strip().split()
                image_path = annotation[0]
                image_name = image_path.split('/')[-1]


                image = cv2.imread(image_path)
                bbox_data_gt = np.array([list(map(float, box.split(','))) for box in annotation[1:]])

                if len(bbox_data_gt) == 0:
                    bboxes_gt=[]
                    classes_gt=[]
                else:
                    bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]

                # print('=> ground truth of %s:' % image_name)
                num_bbox_gt = len(bboxes_gt)
                bboxes_pr = self.predict(image)

                if self.write_image:
                    image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
                    cv2.imwrite(self.write_image_path+image_name, image)

                img_id = imgname2id[image_name]
                imgIds.append(img_id)

                for bbox in bboxes_pr:
                    res_temp = {}
                    coor = np.array(bbox[:4])
                    score = bbox[4]

                    class_ind = int(bbox[5])
                    class_name = self.classes[class_ind]
                    class_ind = classname2id[class_name]
                    xmin, ymin, xmax, ymax = coor
                    w = float(xmax) - float(xmin)
                    h = float(ymax) - float(ymin)

                    res_temp['image_id'] = int(img_id)
                    res_temp['category_id'] = int(class_ind)
                    res_temp['bbox'] = [xmin, ymin, w, h]
                    res_temp['score'] = float(score)

                    result.append(res_temp)
        import json
        if os.path.exists("./results"):
            os.system("rm results/result.json")
        if not os.path.exists("./results"):
            os.mkdir("./results")
        with open("results/result.json", "w") as f:
            json.dump(result, f)
        cocoDt = coco.loadRes("results/result.json")
        annType = "bbox"
        cocoEval = COCOeval(coco, cocoDt, annType)
        print(len(imgIds))
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

def extract_flags():
    flags.DEFINE_integer("batch_size", 8, "batch size for evaluate.")
    flags.DEFINE_integer("image_number", 4942,
                         "number of image for evaluate.")
    flags.DEFINE_string("instances_path", None, "instances json for predictions")
    flags.DEFINE_string("weight_file", None, "path of weight file.")

    return flags.FLAGS

if __name__ == '__main__':
    FLAGS = extract_flags()
    if FLAGS.batch_size:
        cfg.TEST.BATCH_SIZE = FLAGS.batch_size
    if FLAGS.instances_path:
        cfg.TEST.INSTANCES_PATH = FLAGS.instances_path
    if FLAGS.weight_file:
        cfg.TEST.WEIGHT_FILE = FLAGS.weight_file

    YoloTest().evaluate()



