#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 17:50:26
#   Description :
#
#================================================================

import os
import time
import shutil
import numpy as np
import tensorflow as tf
import models.core.utils as utils
from tqdm import tqdm
from models.core.dataset import Dataset
from models.core.yolov3 import YOLOV3
from models.core.config import cfg
from tensorflow.python.ops import variables

flags = tf.flags
FLAGS = None

class YoloTrain(object):
    def __init__(self):
        self.anchor_per_scale    = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes             = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes         = len(self.classes)
        self.learn_rate_init     = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end      = cfg.TRAIN.LEARN_RATE_END
        self.first_stage_epochs  = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods      = cfg.TRAIN.WARMUP_EPOCHS
        self.initial_weight      = cfg.TRAIN.INITIAL_WEIGHT
        self.time                = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay    = cfg.YOLO.MOVING_AVE_DECAY
        self.max_bbox_per_scale  = 150
        self.train_logdir        = "./data/log/train"
        self.trainset            = Dataset('train')
        self.testset             = Dataset('test')
        self.steps_per_period    = len(self.trainset)
        self.start_epoch         = cfg.TRAIN.START_EPOCH

        global_batch_size = FLAGS.batch_size
        if FLAGS.use_horovod:
          global_batch_size *=  hvd.size()

        config = tf.ConfigProto()
        if FLAGS.use_horovod:
            if FLAGS.hvd_device.lower() == "mlu":
                config.mlu_options.visible_device_list=str(hvd.local_rank())
            elif FLAGS.hvd_device.lower() == "gpu":
                config.gpu_options.visible_device_list=str(hvd.local_rank())

        with tf.name_scope('define_input'):
            self.input_data   = tf.placeholder(dtype=tf.float32, name='input_data')
            self.label_sbbox  = tf.placeholder(dtype=tf.float32, name='label_sbbox')
            self.label_mbbox  = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox  = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
            self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
            self.trainable    = tf.placeholder(dtype=tf.bool, name='training')

        with tf.name_scope("define_loss"):
            self.model = YOLOV3(self.input_data, self.trainable)
            self.net_var = tf.global_variables()
            self.giou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(
                                                    self.label_sbbox,  self.label_mbbox,  self.label_lbbox,
                                                    self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
            self.loss = self.giou_loss + self.conf_loss + self.prob_loss

        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                        dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant((self.first_stage_epochs + self.second_stage_epochs)* self.steps_per_period,
                                        dtype=tf.float64, name='train_steps')
            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
                                    (1 + tf.cos(
                                        (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            global_step_update = tf.assign_add(self.global_step, 1.0)

        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())

        with tf.name_scope("define_first_stage_train"):
            self.first_stage_trainable_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                    self.first_stage_trainable_var_list.append(var)

            first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate)
            if FLAGS.use_horovod:
                # if use hvd, lr should be multiplied by the number of cards
                first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate * hvd.size())
                first_stage_optimizer = hvd.DistributedOptimizer(first_stage_optimizer)
            if FLAGS.use_amp:
                first_stage_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(first_stage_optimizer)
            first_stage_optimizer = first_stage_optimizer.minimize(self.loss,
                                                      var_list=self.first_stage_trainable_var_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()

        with tf.name_scope("define_second_stage_train"):
            second_stage_trainable_var_list = tf.trainable_variables()
            second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate)
            if FLAGS.use_horovod:
                # if use hvd, lr should be multiplied by the number of cards
                second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate * hvd.size())
                second_stage_optimizer = hvd.DistributedOptimizer(second_stage_optimizer)
            if FLAGS.use_amp:
                second_stage_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(second_stage_optimizer)
            second_stage_optimizer = second_stage_optimizer.minimize(self.loss,
                                                      var_list=second_stage_trainable_var_list)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()

        self.sess = tf.Session(config=config)

        with tf.name_scope('loader_and_saver'):
            if self.start_epoch == 1:
                var_list = self.net_var
            else:
                var_list = self.net_var + [self.global_step]
            self.loader = tf.train.Saver(var_list)
            self.saver  = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        with tf.name_scope('summary'):
            tf.summary.scalar("learn_rate", self.learn_rate)
            tf.summary.scalar("giou_loss",  self.giou_loss)
            tf.summary.scalar("conf_loss",  self.conf_loss)
            tf.summary.scalar("prob_loss",  self.prob_loss)
            tf.summary.scalar("total_loss", self.loss)

            logdir = "./models/data/log/"
            if (FLAGS.use_horovod and hvd.rank() == 0) or (not FLAGS.use_horovod):
                if os.path.exists(logdir): shutil.rmtree(logdir)
                os.mkdir(logdir)
                self.summary_writer  = tf.summary.FileWriter(logdir, graph=self.sess.graph)
            self.write_op = tf.summary.merge_all()

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        try:
            print('=> Restoring weights from: %s ... ' % self.initial_weight)
            self.loader.restore(self.sess, self.initial_weight)
        except:
            print('=> %s does not exist !!!' % self.initial_weight)
            print('=> Now it starts to train YOLOV3 from scratch ...')
            self.start_epoch = 1
        print("training start from epoch %d."%self.start_epoch)

        if FLAGS.use_horovod:
            self.sess.run(hvd.broadcast_global_variables(0))

        output_dir = FLAGS.output_dir
        if not tf.io.gfile.exists(output_dir):
            if (not FLAGS.use_horovod) or (FLAGS.use_horovod and hvd.rank() == 0):
                tf.io.gfile.makedirs(output_dir)

        run_options = None
        run_metadata = None
        if FLAGS.use_profiler:
            from tensorflow_core.python.client import timeline
            run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            run_metadata = tf.compat.v1.RunMetadata()

        current_step = 0

        if FLAGS.use_performance and (not FLAGS.use_horovod or (FLAGS.use_horovod and hvd.rank() == 0)):
            times = []
        for epoch in range(self.start_epoch, 1+self.first_stage_epochs+self.second_stage_epochs):
            if epoch <= self.first_stage_epochs:
                train_op = self.train_op_with_frozen_variables
            else:
                train_op = self.train_op_with_all_variables
            pbar = tqdm(self.trainset)
            train_epoch_loss, test_epoch_loss = [], []

            if FLAGS.finetune_step and FLAGS.finetune_step > 0:
                k = 1

            for train_data in pbar:
                if FLAGS.finetune_step and FLAGS.finetune_step > 0:
                    if k > FLAGS.finetune_step:
                        break
                    k += 1
                if FLAGS.use_performance and (not FLAGS.use_horovod or (FLAGS.use_horovod and hvd.rank() == 0)):
                  start_time = time.time()

                _, summary, train_step_loss, global_step_val = self.sess.run(
                    [train_op, self.write_op, self.loss, self.global_step], feed_dict={
                                                self.input_data:   train_data[0],
                                                self.label_sbbox:  train_data[1],
                                                self.label_mbbox:  train_data[2],
                                                self.label_lbbox:  train_data[3],
                                                self.true_sbboxes: train_data[4],
                                                self.true_mbboxes: train_data[5],
                                                self.true_lbboxes: train_data[6],
                                                self.trainable:    True,
                }, options=run_options, run_metadata=run_metadata)
                if FLAGS.use_performance:
                  if not FLAGS.use_horovod or (FLAGS.use_horovod and hvd.rank() == 0):
                    times.append(time.time() - start_time)


                current_step += 1

                train_epoch_loss.append(train_step_loss)
                if (not FLAGS.use_horovod) or (FLAGS.use_horovod and hvd.rank() == 0):
                    self.summary_writer.add_summary(summary, global_step_val)
                    pbar.set_description("\n train loss: %.2f \n" %train_step_loss)

            train_epoch_loss = np.mean(train_epoch_loss)
            if (not FLAGS.use_horovod) or (FLAGS.use_horovod and hvd.rank() == 0):
                ckpt_file = FLAGS.output_dir + "/yolov3_train_loss=%.4f.ckpt"%train_epoch_loss
                log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                print("=> Epoch: %2d Time: %s Train loss: %.2f Saving %s ..."
                                %(epoch, log_time, train_epoch_loss, ckpt_file))
                self.saver.save(self.sess, ckpt_file, global_step=epoch)

        if FLAGS.use_profiler:
            if not FLAGS.use_horovod or (FLAGS.use_horovod and hvd.rank() == 0):
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                if not tf.io.gfile.exists("profiler"):
                    tf.io.gfile.mkdir("profiler")
                with open("profiler/timeline.json", 'w') as f:
                    f.write(ctf)
        if FLAGS.use_performance:
          if not FLAGS.use_horovod:
            write_json("summary", FLAGS.batch_size, times)
          elif (FLAGS.use_horovod and hvd.rank() == 0):
            write_json("summary", FLAGS.batch_size*hvd.size(), times)

def extract_flags():
    flags.DEFINE_integer("batch_size", 8,
                         "batch size for training.")
    flags.DEFINE_string("ckpt_path", "",
                        "relative path of ckpt path.")
    flags.DEFINE_integer("first_stage_epochs", 20,
                         "first_stage_epochs for training.")
    flags.DEFINE_integer("second_stage_epochs", 30,
                         "second_stage_epochs for training.")
    flags.DEFINE_integer("start_epoch", 1,
                         "Start Train from which epoch")
    flags.DEFINE_integer("finetune_step", 125,
                         "Just for finetuning test! run finetune_step steps and break anyway.")
    flags.DEFINE_string("output_dir", None,
                        "The output directory where the model checkpoints will be written.")
    flags.DEFINE_bool("use_horovod", False,
                       "use hvd to train nets or not")
    flags.DEFINE_string("hvd_device", None,
                        "The device type when train with horovod.")
    flags.DEFINE_bool("use_profiler", False,
                       "use profiler to train nets or not")
    flags.DEFINE_bool("use_performance", False,
                       "use performance tools to get fps or not")
    flags.DEFINE_bool("use_amp", False,
                       "use automatic mixed precision or not")

    return flags.FLAGS

if __name__ == '__main__':
    FLAGS = extract_flags()
    if FLAGS.batch_size:
        cfg.TRAIN.BATCH_SIZE = FLAGS.batch_size
    if FLAGS.ckpt_path:
        cfg.TRAIN.INITIAL_WEIGHT = FLAGS.ckpt_path
    if FLAGS.first_stage_epochs is not None:
        cfg.TRAIN.FISRT_STAGE_EPOCHS = FLAGS.first_stage_epochs
    if FLAGS.second_stage_epochs is not None:
        cfg.TRAIN.SECOND_STAGE_EPOCHS = FLAGS.second_stage_epochs
    if FLAGS.start_epoch is not None:
        cfg.TRAIN.START_EPOCH = FLAGS.start_epoch
    if FLAGS.use_horovod:
        # import_helper helps to decide whether import hvd or not according to FLAGS.use_horovod.
        from models import import_helper
        global import_helper
        import horovod.tensorflow as hvd
        global hvd
        hvd.init()
        import_helper.add_pkg('hvd', hvd)
    if FLAGS.use_performance:
        from record_time import write_json
        global write_json
    YoloTrain().train()
