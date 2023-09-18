#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

import tensorflow as tf
import horovod.tensorflow as hvd

from utils import image_processing
from utils import hvd_utils

from cambricon import dali
import cambricon.dali.plugin.tf as dali_tf

class HybridPipe(dali.pipeline.Pipeline):

    def __init__(
        self,
        tfrec_filenames,
        tfrec_idx_filenames,
        height,
        width,
        batch_size,
        num_threads,
        device_id,
        shard_id,
        num_mlus,
        deterministic=False,
        dali_cpu=True,
        training=True
    ):

        kwargs = dict()
        if deterministic:
            kwargs['seed'] = 7 * (1 + hvd.rank())
        super(HybridPipe, self).__init__(batch_size, num_threads, device_id, **kwargs)

        self.training = training
        self.input = dali.ops.TFRecordReader(
            path=tfrec_filenames,
            index_path=tfrec_idx_filenames,
            prefetch_queue_depth=12,
            random_shuffle=True,
            shard_id=shard_id,
            num_shards=num_mlus,
            initial_fill=10000,
            features={
                'image/encoded': dali.tfrecord.FixedLenFeature((), dali.tfrecord.string, ""),
                'image/class/label': dali.tfrecord.FixedLenFeature([1], dali.tfrecord.int64, -1),
                'image/class/text': dali.tfrecord.FixedLenFeature([], dali.tfrecord.string, ''),
                'image/object/bbox/xmin': dali.tfrecord.VarLenFeature(dali.tfrecord.float32, 0.0),
                'image/object/bbox/ymin': dali.tfrecord.VarLenFeature(dali.tfrecord.float32, 0.0),
                'image/object/bbox/xmax': dali.tfrecord.VarLenFeature(dali.tfrecord.float32, 0.0),
                'image/object/bbox/ymax': dali.tfrecord.VarLenFeature(dali.tfrecord.float32, 0.0)
            }
        )

        if self.training:
            self.decode = dali.ops.ImageDecoder(device="cpu" if dali_cpu else "mixed", output_type=dali.types.RGB)
            self.resize = dali.ops.RandomResizedCrop(
                device="cpu" if dali_cpu else "gpu",
                size=[width, height],
                interp_type=dali.types.INTERP_LINEAR,
                random_aspect_ratio=[0.75, 1.33],
                random_area=[0.05, 1.0],
                num_attempts=100)
        else:
            # TODO(lvtengda)
            self.decode = dali.ops.ImageDecoder(device="cpu" if dali_cpu else "mixed", output_type=dali.types.RGB)
            # Make sure that every image > 224 for CropMirrorNormalize
            # TODO(lvtengda)
            self.resize = dali.ops.Resize(device="cpu" if dali_cpu else "gpu", resize_shorter=256, subpixel_scale=False, interp_type=dali.types.INTERP_LINEAR)

        self.normalize = dali.ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=dali.types.FLOAT,
            crop=(height, width),
            image_type=dali.types.RGB,
            mean=[123.68, 116.28, 103.53],
            std=[58.395, 57.120, 57.385],
            output_layout=dali.types.NHWC
        )
        self.mirror = dali.ops.CoinFlip()
        self.iter = 0

    def define_graph(self):
        # Read images and labels
        inputs = self.input(name="Reader")
        images = inputs["image/encoded"]
        labels = inputs["image/class/label"].gpu()

        # Decode and augmentation
        images = self.decode(images)
        images = self.resize(images)
        images = self.normalize(images, mirror=self.mirror() if self.training else None)

        return (images, labels)


class DALIPreprocessor(object):

    def __init__(
        self,
        filenames,
        idx_filenames,
        height,
        width,
        batch_size,
        num_threads,
        dtype=tf.uint8,
        dali_cpu=True,
        deterministic=False,
        training=False
    ):
        device_id = hvd.local_rank()
        shard_id = hvd.rank()
        num_mlus = hvd.size()
        pipe = HybridPipe(
            tfrec_filenames=filenames,
            tfrec_idx_filenames=idx_filenames,
            height=height,
            width=width,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            shard_id=shard_id,
            num_mlus=num_mlus,
            deterministic=deterministic,
            dali_cpu=dali_cpu,
            training=training
        )


        daliop = dali_tf.DALIIterator()

        with tf.device("/mlu:0"):
            self.images, self.labels = daliop(
                pipeline=pipe,
                shapes=[(batch_size, height, width, 3), (batch_size, 1)],
                dtypes=[tf.float32, tf.int64],
                device_id=device_id
            )

    def get_device_minibatches(self):
        with tf.device("/mlu:0"):
            self.labels -= 1  # Change to 0-based (don't use background class)
            self.labels = tf.squeeze(self.labels, axis=-1)
        return self.images, self.labels
