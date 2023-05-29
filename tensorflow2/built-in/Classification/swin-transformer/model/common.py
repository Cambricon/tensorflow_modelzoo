# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Common util functions and classes used by both keras cifar and imagenet."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
from absl import logging
import tensorflow as tf

import tensorflow_model_optimization as tfmot
import model.core as flags_core
from model import keras_utils

TRAIN_TOP_1 = 'training_accuracy_top_1'

def get_optimizer(learning_rate=1e-5, epsilon=1e-8):
  """Returns optimizer to use."""
  # The learning_rate is overwritten at the beginning of each step by callback.
  return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, epsilon=epsilon)


def get_callbacks(FLAGS,
                  pruning_method=None,
                  enable_checkpoint_and_export=False,
                  model_dir=None,
                  use_profiler=False):
  """Returns common callbacks."""
  if FLAGS.model_dir is not None:
    tf.io.gfile.makedirs(FLAGS.model_dir)

  time_callback = keras_utils.TimeHistory(
      FLAGS.batch_size,
      FLAGS.log_steps,
      logdir=FLAGS.model_dir if FLAGS.use_profiler else None)
  callbacks = [time_callback]

  if use_profiler:
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=FLAGS.model_dir, profile_batch=(1, FLAGS.finetune_steps-1))
    callbacks.append(tensorboard_callback)

  is_pruning_enabled = pruning_method is not None
  if is_pruning_enabled:
    callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())
    if model_dir is not None:
      callbacks.append(
          tfmot.sparsity.keras.PruningSummaries(
              log_dir=model_dir, profile_batch=0))

  if enable_checkpoint_and_export:
    if model_dir is not None:
      ckpt_full_path = os.path.join(model_dir, 'model.ckpt-{epoch:04d}')
      callbacks.append(
          tf.keras.callbacks.ModelCheckpoint(
              ckpt_full_path, save_weights_only=True))
  return callbacks


def build_stats(history, eval_output, callbacks):
  """Normalizes and returns dictionary of stats.

  Args:
    history: Results of the training step. Supports both categorical_accuracy
      and sparse_categorical_accuracy.
    eval_output: Output of the eval step. Assumes first value is eval_loss and
      second value is accuracy_top_1.
    callbacks: a list of callbacks which might include a time history callback
      used during keras.fit.

  Returns:
    Dictionary of normalized results.
  """
  stats = {}
  if eval_output:
    stats['accuracy_top_1'] = float(eval_output[1])
    stats['eval_loss'] = float(eval_output[0])
  if history and history.history:
    train_hist = history.history
    # Gets final loss from training.
    stats['loss'] = float(train_hist['loss'][-1])
    # Gets top_1 training accuracy.
    if 'categorical_accuracy' in train_hist:
      stats[TRAIN_TOP_1] = float(train_hist['categorical_accuracy'][-1])
    elif 'sparse_categorical_accuracy' in train_hist:
      stats[TRAIN_TOP_1] = float(train_hist['sparse_categorical_accuracy'][-1])
    elif 'accuracy' in train_hist:
      stats[TRAIN_TOP_1] = float(train_hist['accuracy'][-1])

  if not callbacks:
    return stats

  # Look for the time history callback which was used during keras.fit
  for callback in callbacks:
    if isinstance(callback, keras_utils.TimeHistory):
      timestamp_log = callback.timestamp_log
      stats['step_timestamp_log'] = timestamp_log
      stats['train_finish_time'] = callback.train_finish_time
      if callback.epoch_runtime_log:
        stats['avg_exp_per_second'] = callback.average_examples_per_second

  return stats


def define_keras_flags(model=False,
                       optimizer=False,
                       pretrained_filepath=False):
  """Define flags for Keras models."""
  flags_core.define_base(
      clean=True,
      num_gpu=True,
      run_eagerly=True,
      train_epochs=True,
      epochs_between_evals=True,
      distribution_strategy=True)
  flags_core.define_performance(
      num_parallel_calls=False,
      inter_op=True,
      intra_op=True,
      synthetic_data=True,
      dtype=True,
      all_reduce_alg=True,
      num_packs=True,
      tf_gpu_thread_mode=True,
      datasets_num_private_threads=True,
      loss_scale=True,
      fp16_implementation=True,
      tf_data_experimental_slack=True,
      enable_xla=True,
      training_dataset_cache=True)
  flags_core.define_image()
  flags_core.define_benchmark()
  flags_core.define_distribution()
  flags.adopt_module_key_flags(flags_core)

  flags.DEFINE_boolean(name='enable_eager', default=False, help='Enable eager?')
  flags.DEFINE_boolean(name='skip_eval', default=False, help='Skip evaluation?')
  # TODO(b/135607288): Remove this flag once we understand the root cause of
  # slowdown when setting the learning phase in Keras backend.
  flags.DEFINE_boolean(
      name='set_learning_phase_to_train',
      default=True,
      help='If skip eval, also set Keras learning phase to 1 (training).')
  flags.DEFINE_boolean(
      name='explicit_gpu_placement',
      default=False,
      help='If not using distribution strategy, explicitly set device scope '
      'for the Keras training loop.')
  flags.DEFINE_boolean(
      name='use_trivial_model',
      default=False,
      help='Whether to use a trivial Keras model.')
  flags.DEFINE_boolean(
      name='report_accuracy_metrics',
      default=True,
      help='Report metrics during training and evaluation.')
  flags.DEFINE_boolean(
      name='use_tensor_lr',
      default=True,
      help='Use learning rate tensor instead of a callback.')
  flags.DEFINE_boolean(
      name='use_profiler',
      default=False,
      help='Whether to enable Tensorboard callback.')
  flags.DEFINE_integer(
      name='train_steps',
      default=None,
      help='The number of steps to run for training. If it is larger than '
      '# batches per epoch, then use # batches per epoch. This flag will be '
      'ignored if train_epochs is set to be larger than 1. ')
  flags.DEFINE_boolean(
      name='batchnorm_spatial_persistent',
      default=True,
      help='Enable the spacial persistent mode for CuDNN batch norm kernel.')
  flags.DEFINE_boolean(
      name='enable_get_next_as_optional',
      default=False,
      help='Enable get_next_as_optional behavior in DistributedIterator.')
  flags.DEFINE_boolean(
      name='enable_checkpoint_and_export',
      default=False,
      help='Whether to enable a checkpoint callback and export the savedmodel.')
  flags.DEFINE_string(name='tpu', default='', help='TPU address to connect to.')
  flags.DEFINE_integer(
      name='steps_per_loop',
      default=None,
      help='Number of steps per training loop. Only training step happens '
      'inside the loop. Callbacks will not be called inside. Will be capped at '
      'steps per epoch.')
  flags.DEFINE_boolean(
      name='use_tf_while_loop',
      default=True,
      help='Whether to build a tf.while_loop inside the training loop on the '
      'host. Setting it to True is critical to have peak performance on '
      'TPU.')

  if model:
    flags.DEFINE_string('model', 'resnet50_v1.5',
                        'Name of model preset. (mobilenet, resnet50_v1.5)')
  if optimizer:
    flags.DEFINE_string(
        'optimizer', 'resnet50_default', 'Name of optimizer preset. '
        '(mobilenet_default, resnet50_default)')
    # TODO(kimjaehong): Replace as general hyper-params not only for mobilenet.
    flags.DEFINE_float(
        'initial_learning_rate_per_sample', 0.00007,
        'Initial value of learning rate per sample for '
        'mobilenet_default.')
    flags.DEFINE_float('lr_decay_factor', 0.94,
                       'Learning rate decay factor for mobilenet_default.')
    flags.DEFINE_float('num_epochs_per_decay', 2.5,
                       'Number of epochs per decay for mobilenet_default.')
  if pretrained_filepath:
    flags.DEFINE_string('pretrained_filepath', '', 'Pretrained file path.')


def get_synth_data(height, width, num_channels, num_classes, dtype):
  """Creates a set of synthetic random data.

  Args:
    height: Integer height that will be used to create a fake image tensor.
    width: Integer width that will be used to create a fake image tensor.
    num_channels: Integer depth that will be used to create a fake image tensor.
    num_classes: Number of classes that should be represented in the fake labels
      tensor
    dtype: Data type for features/images.

  Returns:
    A tuple of tensors representing the inputs and labels.

  """
  # Synthetic input should be within [0, 255].
  inputs = tf.random.truncated_normal([height, width, num_channels],
                                      dtype=dtype,
                                      mean=127,
                                      stddev=60,
                                      name='synthetic_inputs')
  labels = tf.random.uniform([1],
                             minval=0,
                             maxval=num_classes - 1,
                             dtype=tf.int32,
                             name='synthetic_labels')
  return inputs, labels


def define_pruning_flags():
  """Define flags for pruning methods."""
  flags.DEFINE_string(
      'pruning_method', None, 'Pruning method.'
      'None (no pruning) or polynomial_decay.')
  flags.DEFINE_float('pruning_initial_sparsity', 0.0,
                     'Initial sparsity for pruning.')
  flags.DEFINE_float('pruning_final_sparsity', 0.5,
                     'Final sparsity for pruning.')
  flags.DEFINE_integer('pruning_begin_step', 0, 'Begin step for pruning.')
  flags.DEFINE_integer('pruning_end_step', 100000, 'End step for pruning.')
  flags.DEFINE_integer('pruning_frequency', 100, 'Frequency for pruning.')


def define_clustering_flags():
  """Define flags for clustering methods."""
  flags.DEFINE_string('clustering_method', None,
                      'None (no clustering) or selective_clustering '
                      '(cluster last three Conv2D layers of the model).')


def get_synth_input_fn(height,
                       width,
                       num_channels,
                       num_classes,
                       dtype=tf.float32,
                       drop_remainder=True):
  """Returns an input function that returns a dataset with random data.

  This input_fn returns a data set that iterates over a set of random data and
  bypasses all preprocessing, e.g. jpeg decode and copy. The host to device
  copy is still included. This used to find the upper throughput bound when
  tuning the full input pipeline.

  Args:
    height: Integer height that will be used to create a fake image tensor.
    width: Integer width that will be used to create a fake image tensor.
    num_channels: Integer depth that will be used to create a fake image tensor.
    num_classes: Number of classes that should be represented in the fake labels
      tensor
    dtype: Data type for features/images.
    drop_remainder: A boolean indicates whether to drop the remainder of the
      batches. If True, the batch dimension will be static.

  Returns:
    An input_fn that can be used in place of a real one to return a dataset
    that can be used for iteration.
  """

  # pylint: disable=unused-argument
  def input_fn(is_training, data_dir, batch_size, *args, **kwargs):
    """Returns dataset filled with random data."""
    inputs, labels = get_synth_data(
        height=height,
        width=width,
        num_channels=num_channels,
        num_classes=num_classes,
        dtype=dtype)
    # Cast to float32 for Keras model.
    labels = tf.cast(labels, dtype=tf.float32)
    data = tf.data.Dataset.from_tensors((inputs, labels)).repeat()

    # `drop_remainder` will make dataset produce outputs with known shapes.
    data = data.batch(batch_size, drop_remainder=drop_remainder)
    data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return data

  return input_fn


def set_thread_mode_and_count(mlu_thread_mode, per_mlu_thread_count,
                              interop_threads, intraop_threads):
  os.environ['TF_MLU_THREAD_MODE'] = mlu_thread_mode
  os.environ['TF_MLU_THREAD_COUNT'] = str(per_mlu_thread_count)
  os.environ['TF_NUM_INTEROP_THREADS'] = str(interop_threads)
  os.environ['TF_NUM_INTRAOP_THREADS'] = str(intraop_threads)
  logging.info('TF_MLU_THREAD_COUNT: %s', os.environ['TF_MLU_THREAD_COUNT'])
  logging.info('TF_MLU_THREAD_MODE: %s', os.environ['TF_MLU_THREAD_MODE'])
  logging.info('TF_NUM_INTEROP_THREADS: %s', os.environ['TF_NUM_INTEROP_THREADS'])
  logging.info('TF_NUM_INTRAOP_THREADS: %s', os.environ['TF_NUM_INTRAOP_THREADS'])
