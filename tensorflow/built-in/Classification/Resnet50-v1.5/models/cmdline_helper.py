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

import argparse

import tensorflow as tf

def parse_cmdline(available_arch):
    tf.app.flags.DEFINE_boolean('use_amp', False, """If use amp, please set True.""")
    tf.app.flags.DEFINE_string(
        'mode',
        'train_and_evaluate',
        "The execution mode of the script."
        "choices=['train', 'train_and_evaluate', 'evaluate', 'predict', 'training_benchmark', 'inference_benchmark']"
    )
    tf.app.flags.DEFINE_string(
        'data_dir',
        '/data/tensorflowtraining/datasets/ILSVRC2012/ilsvrc12_tfrecord/',
        "Path to dataset in TFRecord format. Files should be named 'train-*' and 'validation-*'."
    )
    tf.app.flags.DEFINE_string(
        'data_idx_dir',
        '/data/tensorflow/training/datasets/dali_index_dir/',
        "Path to index files for DALI. Files should be named 'train-*' and 'validation-*'."
    )
    tf.app.flags.DEFINE_string('export_dir', "./export_dir", "Directory in which to write exported SavedModel.")
    tf.app.flags.DEFINE_string('to_predict', None, "Path to file or directory of files to run prediction on.")
    tf.app.flags.DEFINE_integer('batch_size', 128, """Size of each minibatch per MLU.""")
    tf.app.flags.DEFINE_integer('eval_steps', 0, """Number of eval to run.""")
    tf.app.flags.DEFINE_integer('num_iter', 90, """Number of iterations to run.""")
    tf.app.flags.DEFINE_integer('run_iter', -1, """Number of training iterations to run on single run.""")
    tf.app.flags.DEFINE_string('iter_unit', 'epoch', "Unit of iterations, choices=['epoch', 'batch'].")
    tf.app.flags.DEFINE_integer(
        'warmup_steps',
        100,
        """Number of steps considered as warmup and not taken into account for performance measurements."""
    )

    # Tensor format used for the computation.
    tf.app.flags.DEFINE_string('data_format', 'NHWC', "data format, choices=['NHWC', 'NCHW'].")
    tf.app.flags.DEFINE_string(
        'model_dir',
        "mlu_model",
        """Directory in which to write model. If undefined, results dir will be used."""
    )
    tf.app.flags.DEFINE_string(
        'results_dir',
        '.',
        """Directory in which to write training logs, summaries and checkpoints."""
    )
    tf.app.flags.DEFINE_string(
        'finetune_checkpoint',
        None,
        "Path to pre-trained checkpoint which will be used for fine-tuning"
    )
    tf.app.flags.DEFINE_boolean(
        "use_final_conv",
        False,
        "Use cosine learning rate schedule."
    )
    tf.app.flags.DEFINE_integer('quant_delay', 0, "Number of steps to be run before quantization starts to happen")
    tf.app.flags.DEFINE_boolean(
        "quantize",
        False,
        "Quantize weights and activations during training. (Defaults to Assymmetric quantization)"
    )
    tf.app.flags.DEFINE_boolean(
        "use_qdq",
        False,
        "Use QDQV3 op instead of FakeQuantWithMinMaxVars op for quantization. QDQv3 does only scaling"
    )
    tf.app.flags.DEFINE_boolean(
        "symmetric",
        False,
        "Quantize weights and activations during training using symmetric quantization."
    )
    tf.app.flags.DEFINE_string('log_filename', 'log.json', "Name of the JSON file to which write the training log")
    tf.app.flags.DEFINE_integer('display_every', 1, """How often (in batches) to print out running information.""")
    tf.app.flags.DEFINE_float('lr_init', 0.256, """Initial value for the learning rate.""")
    tf.app.flags.DEFINE_integer('lr_warmup_epochs', 8, """Number of warmup epochs for learning rate schedule.""")
    tf.app.flags.DEFINE_float('weight_decay', 3.0517578125e-05, """Weight Decay scale factor.""")
    tf.app.flags.DEFINE_string(
        'weight_init',
        'fan_in',
        """Model weight initialization method, choices=['fan_in', 'fan_out'].""")
    tf.app.flags.DEFINE_float('momentum', 0.875, """SGD momentum value for the Momentum optimizer.""")

    # Select fp32 or non-AMP fp16 precision arithmetic.
    tf.app.flags.DEFINE_string('precision', 'fp32', " choices=['fp32', 'fp16'].")
    tf.app.flags.DEFINE_float('loss_scale', 256.0, """Loss scale for FP16 Training and Fast Math FP32.""")
    tf.app.flags.DEFINE_float('label_smoothing', 0.1, """The value of label smoothing.""")
    tf.app.flags.DEFINE_float('mixup', 0.0, """The alpha parameter for mixup (if 0 then mixup is not applied).""")
    tf.app.flags.DEFINE_boolean('use_horovod', False, """If use horovod, please set True.""")
    tf.app.flags.DEFINE_boolean('use_profiler', False, """If use profile tools, please set True.""")
    tf.app.flags.DEFINE_boolean('use_performance', False, """If use performance test tools, please set True.""")
    tf.app.flags.DEFINE_boolean('hvd_finetune_test', False, "Whether to run hvd finetune from 1card trained ckpt")
    tf.app.flags.DEFINE_boolean("use_static_loss_scaling", False, "Use static loss scaling in FP16 or FP32 AMP.")
    tf.app.flags.DEFINE_boolean(
        "use_xla",
        False,
        "Enable XLA (Accelerated Linear Algebra) computation for improved performance.")
    tf.app.flags.DEFINE_boolean("use_dali", False, "Enable DALI data input.")
    tf.app.flags.DEFINE_boolean(
        "use_tf_amp",
        False,
        "Enable Automatic Mixed Precision to speedup FP32 computation using tensor cores.")
    tf.app.flags.DEFINE_boolean("use_cosine_lr", False, "Use cosine learning rate schedule.")
    tf.app.flags.DEFINE_integer('seed', None, """Random seed.""")
    tf.app.flags.DEFINE_float(
        'device_memory_fraction',
        0.7,
        """Limit memory fraction used by training script for DALI""")
    tf.app.flags.DEFINE_integer(
        'device_id',
        0,
        """Specify ID of the target device on multi-device platform. Effective only for single-device mode.""")
    tf.app.flags.DEFINE_string('device', 'MLU', """Use device to deploy clones.""")

    FLAGS = tf.app.flags.FLAGS

    return FLAGS
