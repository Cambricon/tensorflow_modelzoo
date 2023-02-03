# Lint as: python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Runs an Image Classification model."""

import os, sys
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils
from model import callbacks as custom_callbacks
from model.model import CFGS, SwinTransformer  
from model import distribution_utils
from model import common
from model import imagenet_preprocessing


def _get_metrics(one_hot):
    if one_hot:
        return {
            'acc': tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            'accuracy': tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            'top_1': tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            'top_5': tf.keras.metrics.TopKCategoricalAccuracy(
                k=5,
                name='top_5_accuracy'),
        }
    else:
        return {
            'acc': tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            'accuracy': tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            'top_1': tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            'top_5': tf.keras.metrics.SparseTopKCategoricalAccuracy(
                 k=5,
                 name='top_5_accuracy'),
         }

def get_dataset(is_training, data_dir, batch_size, use_sync=False, dtype=tf.float32, hvd=None):
    if use_sync:
        return imagenet_preprocessing.get_synth_input_fn(
          height=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
          width=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
          num_channels=imagenet_preprocessing.NUM_CHANNELS,
          num_classes=imagenet_preprocessing.NUM_CLASSES,
          dtype=dtype,
          drop_remainder=True)
    else:
        return imagenet_preprocessing.input_fn(is_training, data_dir, batch_size, dtype, hvd=hvd)


def resume_from_checkpoint(model, model_dir, train_steps, checkpoint):
  """Resumes from the latest checkpoint, if possible.

  Loads the model weights and optimizer settings from a checkpoint.
  This function should be used in case of preemption recovery.

  Args:
    model: The model whose weights should be restored.
    model_dir: The directory where model weights were saved.
    train_steps: The number of steps to train.

  Returns:
    The epoch of the latest checkpoint, or 0 if not restoring.

  """
  logging.info('Load from checkpoint is enabled.')
  if checkpoint:
    model.load_weights(checkpoint)
    print("load checkpoint:", checkpoint, " Success!")
  else:
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    logging.info('latest_checkpoint: %s', latest_checkpoint)
    if not latest_checkpoint:
      logging.info('No checkpoint detected.')
      return 0

    logging.info('Checkpoint file %s found and restoring from '
               'checkpoint', latest_checkpoint)
    model.load_weights(latest_checkpoint)
  initial_epoch = model.optimizer.iterations // int(train_steps)
  logging.info('Completed loading from checkpoint.')
  return int(initial_epoch)


def define_classifier_flags():
    """Defines common flags for image classification."""
    flags.DEFINE_string(
        'data_dir',
        default=None,
        help='The location of the input data.')
    flags.DEFINE_string(
        'mode',
        default=None,
        help='Mode to run: `train`, `eval`, `train_and_eval` or `export`.')
    flags.DEFINE_bool(
        'run_eagerly',
        default=None,
        help='Use eager execution and disable autograph for debugging.')
    flags.DEFINE_string(
        'dataset',
        default=None,
        help='The name of the dataset, e.g. ImageNet, etc.')
    flags.DEFINE_integer(
        'log_steps',
        default=100,
        help='The interval of steps between logging of batch level stats.')
    flags.DEFINE_string(
        'model_dir',
        default="mlu_model",
        help='The path for storage of checkpoints in training.')
    flags.DEFINE_string(
        'model_name',
        default="swin_large_224",
        help='The name for swin-transformer pretrained model.')
    flags.DEFINE_integer(
        'num_mlus',
        default=1,
        help='The number of mlu devices with natively distributed.')
    flags.DEFINE_integer(
        'num_gpus',
        default=1,
        help='The number of gpu devices with natively distributed.')
    flags.DEFINE_integer(
        'batch_size',
        default=None,
        help='The number of batch_size.')
    flags.DEFINE_integer(
        'profile_steps',
        default=1000,
        help='The interval steps of profile in training.')
    flags.DEFINE_integer(
        'finetune_steps',
        default=None,
        help='Assign finetune steps when run less than 1 epoch.')
    flags.DEFINE_bool(
        'use_horovod',
        default=False,
        help='Use hvd to train networks.')
    flags.DEFINE_bool(
        'use_profiler',
        default=False,
        help='If use tensorboard, please set True.')
    flags.DEFINE_bool(
        'use_performance',
        default=False,
        help='Use performance.')
    flags.DEFINE_bool(
        'use_dummy_synthetic_data',
        default=False,
        help='Use dummy synthetic data.')
    flags.DEFINE_integer(
        'validation_steps',
        default=None,
        help='Assign validation_steps steps when use train_and_eval.')
    flags.DEFINE_float(
        'lr',
        default=1e-5,
        help='learning rate for Adam Optimizer.')
    flags.DEFINE_float(
        'epsilon',
        default=1e-8,
        help='The epsilon for Adam Optimizer.')
    flags.DEFINE_integer(
        'epochs',
        default=None,
        help='The number of epochs.')
    flags.DEFINE_bool(
        'skip_eval',
        default=False,
        help='Whether to skip_eval per epoch')
    flags.DEFINE_bool(
        'one_hot',
        default=False,
        help='is or not one hot')
    flags.DEFINE_bool(
        'use_amp',
        default=False,
        help='is or not use amp mode')
    flags.DEFINE_string(
      'distribution_strategy',
      default=None,
      help="Use which distribution_strategy to train. eg. 'off', 'mlu', 'MirroredStrategy' etc." )
    flags.DEFINE_string(
      'checkpoint_file',
      default=None,
      help="Loading checkpoint file for evaluation!" )
    flags.DEFINE_string(
      'finetune_checkpoint',
      default=None,
      help="Incoming the path of ckpt. " )
    flags.DEFINE_string(
      'initialize_checkpoint',
      default=None,
      help="Initializer the path of ckpt. " )


def build_model(model_name, pretrained, pretrained_ckpt=None, use_hvd=False, horovod=None):
    img_adjust_layer = tf.keras.layers.Lambda(lambda data: 
             tf.keras.applications.imagenet_utils.preprocess_input(
                tf.cast(data, tf.float32), mode="tf"), input_shape=[224, 224, 3])

    pretrained_model = SwinTransformer(model_name, num_classes=imagenet_preprocessing.NUM_CLASSES,
                       include_top=False, pretrained=pretrained, pretrained_checkpoint=pretrained_ckpt, 
                       use_hvd=use_hvd, horovod=horovod)

    model = tf.keras.Sequential([
        img_adjust_layer,
        pretrained_model,
        tf.keras.layers.Dense(imagenet_preprocessing.NUM_CLASSES, activation='softmax')
    ])

    return model

def eval(params):
    """Runs the eval path using compile/evaluate."""
    logging.info('Running evaluation.')

    if params.use_performance and params.use_profiler:
        raise ValueError("You can only set use_profiler or use_performance, not at the same time, otherwise the e2e time will be worse")
    
    if params.use_profiler:
        from tensorflow.python.client import timeline

    if params.use_performance:
        from record_time import TimeHistoryRecord, write_json
        global TimeHistoryRecord
        global write_json

    validation_dataset = get_dataset(is_training=False, data_dir=params.data_dir,
            batch_size=params.batch_size, use_sync=params.use_dummy_synthetic_data)

    if params.model_name in CFGS.keys():
        model_name = params.model_name 
    else:
        raise Warning("Not Found Model!")

    pretrained_ckpt = params.initialize_checkpoint
    pretrained = params.finetune_checkpoint == "" and params.initialize_checkpoint == ""

    train_steps = imagenet_preprocessing.NUM_IMAGES["train"] / params.batch_size
    validation_steps = imagenet_preprocessing.NUM_IMAGES["validation"] / params.batch_size

    model = build_model(model_name, pretrained, pretrained_ckpt)

    metrics_map = _get_metrics(params.one_hot)
    metrics = [metrics_map[metric] for metric in ["top_1", "top_5"]]
    
    if params.one_hot:
        loss_obj = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=params.model.loss.label_smoothing,
            reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE)
    else:
        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()

    run_metadata = None
    if params.use_profiler:
        run_metadata = tf.compat.v1.RunMetadata()

    optimizer = common.get_optimizer(learning_rate=params.lr, epsilon=params.epsilon)
    model.compile(
        optimizer=optimizer,
        loss=loss_obj,
        metrics=metrics,
    )

    if params.checkpoint_file is not None:
        resume_from_checkpoint(model=model,
                               model_dir=params.model_dir,
                               train_steps=train_steps,
                               checkpoint=params.checkpoint_file)
    
    validation_output = model.evaluate(
        validation_dataset, steps=validation_steps, verbose=2)
    
    print('Run stats:\ntop_1 acc: %s\t eval_loss: %s'
                              %(validation_output[1], validation_output[0]))


def run(flags_obj):
    """Runs Image Classification model using native Keras APIs.

    Args:
      flags_obj: An object containing parsed flag values.
      strategy_override: A `tf.distribute.Strategy` object to use for model.

    Returns:
      Dictionary of training/eval stats
    """
    if flags_obj.mode == 'eval':
      eval(flags_obj)
    else:
      raise ValueError('{} is not a valid mode.'.format(flags_obj.mode))


def main(_):
    stats = run(flags.FLAGS)
    if stats and ((flags.FLAGS.use_horovod and hvd.rank() == 0) or not flags.FLAGS.use_horovod):
        logging.info('Run stats:\n%s', stats)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    define_classifier_flags()
    flags.mark_flag_as_required('data_dir')
    flags.mark_flag_as_required('mode')
    flags.mark_flag_as_required('dataset')
    
    app.run(main)
