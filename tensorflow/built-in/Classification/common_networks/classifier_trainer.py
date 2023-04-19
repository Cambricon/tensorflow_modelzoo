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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint
from typing import Any, Tuple, Text, Optional, Mapping

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils
from models.modeling import performance
from models.modeling.hyperparams import params_dict
from models.utils import hyperparams_flags
from models.utils.logs import logger
from models.utils.misc import distribution_utils
from models.utils.misc import keras_utils
from models.vision import callbacks as custom_callbacks
from models.vision import dataset_factory
from models.vision import optimizer_factory
from models.vision.configs import base_configs
from models.vision.configs import configs
from models.vision.inceptionv3 import common
from models.vision.resnet import resnet_model
from models.vision.inceptionv3 import inceptionv3_model
from models.vision.vgg import vgg_model
from models.vision.resnet import resnet101_model
from models.vision.mobilenet import mobilenetv2_model
from models.vision.densenet import densenet_model


def get_models() -> Mapping[str, tf.keras.Model]:
  """Returns the mapping from model type name to Keras model."""
  return  {
      'resnet': resnet_model.resnet50,
      'inceptionv3': inceptionv3_model.inceptionv3,
      'resnet50': resnet_model.resnet50,
      'vgg19': vgg_model.vgg19,
      'resnet101': resnet101_model.resnet101,
      'mobilenetv2': mobilenetv2_model.mobilenetv2,
      'densenet201':densenet_model.densenet201,
  }

def _get_metrics(one_hot: bool) -> Mapping[Text, Any]:
  """Get a dict of available metrics to track."""
  if one_hot:
    return {
        # (name, metric_fn)
        'acc': tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        'accuracy': tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        'top_1': tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        'top_5': tf.keras.metrics.TopKCategoricalAccuracy(
            k=5,
            name='top_5_accuracy'),
    }
  else:
    return {
        # (name, metric_fn)
        'acc': tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        'accuracy': tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        'top_1': tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        'top_5': tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=5,
            name='top_5_accuracy'),
    }


def get_image_size_from_model(
    params: base_configs.ExperimentConfig) -> Optional[int]:
  """If the given model has a preferred image size, return it."""
  return None


def _get_dataset_builders(params: base_configs.ExperimentConfig,
                          strategy: tf.distribute.Strategy,
                          one_hot: bool
                         ) -> Tuple[Any, Any]:
  """Create and return train and validation dataset builders."""
  if one_hot:
    logging.warning('label_smoothing > 0, so datasets will be one hot encoded.')
  else:
    logging.warning('label_smoothing not applied, so datasets will not be one '
                    'hot encoded.')

  num_devices = strategy.num_replicas_in_sync if strategy else 1

  image_size = get_image_size_from_model(params)

  dataset_configs = [
      params.train_dataset, params.validation_dataset
  ]
  builders = []

  for config in dataset_configs:
    if params.use_dummy_synthetic_data:
      config.builder = 'synthetic'
    if config is not None and config.has_data:
      if flags.FLAGS.batch_size == 0:
         builder = dataset_factory.DatasetBuilder(
          config,
          image_size=image_size or config.image_size,
          num_devices=num_devices,
          one_hot=one_hot)
      else:
        builder = dataset_factory.DatasetBuilder(
            config,
            batch_size=flags.FLAGS.batch_size,
            image_size=image_size or config.image_size,
            num_devices=num_devices,
            one_hot=one_hot)
    else:
      builder = None
    builders.append(builder)

  return builders


def get_loss_scale(params: base_configs.ExperimentConfig,
                   fp16_default: float = 128.) -> float:
  """Returns the loss scale for initializations."""
  loss_scale = params.runtime.loss_scale
  if loss_scale == 'dynamic':
    return loss_scale
  elif loss_scale is not None:
    return float(loss_scale)
  elif params.train_dataset.dtype == 'float32':
    return 1.
  else:
    assert params.train_dataset.dtype == 'float16'
    return fp16_default


def _get_params_from_flags(flags_obj: flags.FlagValues):
  """Get ParamsDict from flags."""
  model = flags_obj.model_type.lower()
  dataset = flags_obj.dataset.lower()
  params = configs.get_config(model=model, dataset=dataset)

  flags_overrides = {
      'model_dir': flags_obj.model_dir,
      'batch_size': flags_obj.batch_size,
      'epochs': flags_obj.epochs,
      'start_epoch': flags_obj.start_epoch,
      'mode': flags_obj.mode,
      'use_horovod': flags_obj.use_horovod,
      'use_profiler': flags_obj.use_profiler,
      'use_performance': flags_obj.use_performance,
      'use_dummy_synthetic_data': flags_obj.use_dummy_synthetic_data,
      'use_amp': flags_obj.use_amp,
      'use_qat': flags_obj.use_qat,
      'inter_op_threads': flags_obj.inter_op_threads,
      'intra_op_threads': flags_obj.intra_op_threads,
      'finetune_steps': flags_obj.finetune_steps,
      'validation_steps': flags_obj.validation_steps,
      'finetune_checkpoint' : flags_obj.finetune_checkpoint,
      'model': {
          'name': model,
      },
      'runtime': {
          'run_eagerly': flags_obj.run_eagerly,
          'tpu': flags_obj.tpu,
          'num_mlus': flags_obj.num_mlus,
          'num_gpus': flags_obj.num_gpus,
      },
      'train_dataset': {
          'data_dir': flags_obj.data_dir,
      },
      'validation_dataset': {
          'data_dir': flags_obj.data_dir,
      },
      'train': {
          'time_history': {
              'log_steps': flags_obj.log_steps,
          },
      },
      'evaluation': {
          'skip_eval': flags_obj.skip_eval
      }
  }

  if flags_obj.distribution_strategy:
    flags_overrides['runtime']['distribution_strategy'] = flags_obj.distribution_strategy

  overriding_configs = (flags_obj.config_file,
                        flags_obj.params_override,
                        flags_overrides)

  pp = pprint.PrettyPrinter()

  logging.info('Base params: %s', pp.pformat(params.as_dict()))

  for param in overriding_configs:
    logging.info('Overriding params: %s', param)
    # Set is_strict to false because we can have dynamic dict parameters.
    params = params_dict.override_params_dict(params, param, is_strict=False)

  params.validate()
  params.lock()

  logging.info('Final model parameters: %s', pp.pformat(params.as_dict()))
  return params


def resume_from_checkpoint(model: tf.keras.Model,
                           model_dir: str,
                           train_steps: int,
                           finetune_checkpoint: str,
                           params: base_configs.ExperimentConfig) -> int:
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
  if finetune_checkpoint:
    model.load_weights(finetune_checkpoint)
  else:
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    logging.info('latest_checkpoint: %s', latest_checkpoint)

    if not latest_checkpoint:
      logging.info('No checkpoint detected.')
      return 0

    logging.info('Checkpoint file %s found and restoring from '
                'checkpoint', latest_checkpoint)
    model.load_weights(latest_checkpoint)
  sess = tf.keras.backend.get_session()
  iterations = sess.run(model.optimizer.iterations)
  initial_epoch = params.start_epoch or iterations // train_steps
  logging.info('Completed loading from checkpoint.')
  logging.info('Resuming from epoch %d', initial_epoch)
  return int(initial_epoch)


def initialize(params: base_configs.ExperimentConfig,
               dataset_builder: dataset_factory.DatasetBuilder):
  """Initializes backend related initializations."""
  data_format = 'channels_last'
  tf.keras.backend.set_image_data_format(data_format)

  session_config = tf.ConfigProto()

  if params.use_horovod:
    if params.runtime.num_gpus > 0:
      session_config.gpu_options.visible_device_list = str(hvd.local_rank())
    elif params.runtime.num_mlus > 0:
      session_config.mlu_options.visible_device_list = str(hvd.local_rank())

  session_config.allow_soft_placement=True
  session_config.log_device_placement=False
  tf.keras.backend.set_session(tf.Session(config=session_config))

def define_classifier_flags():
  """Defines common flags for image classification."""
  hyperparams_flags.initialize_common_flags()
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
      'model_type',
      default=None,
      help='The type of the model, e.g. EfficientNet, etc.')
  flags.DEFINE_string(
      'dataset',
      default=None,
      help='The name of the dataset, e.g. ImageNet, etc.')
  flags.DEFINE_integer(
      'log_steps',
      default=100,
      help='The interval of steps between logging of batch level stats.')
  flags.DEFINE_integer(
      'num_mlus',
      default=1,
      help='The number of mlu devices with natively distributed.')
  flags.DEFINE_integer(
      'batch_size',
      default=None,
      help='The number of batch_size.')
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
       help='Use profiler.')
  flags.DEFINE_bool(
       'use_performance',
       default=False,
       help='Use performance.')
  flags.DEFINE_bool(
       'use_dummy_synthetic_data',
       default=False,
       help='Use dummy synthetic data.')
  flags.DEFINE_bool(
       'use_amp',
       default=False,
       help='Use amp.')
  flags.DEFINE_bool(
       'use_qat',
       default=False,
       help='Use quantization aware training.')
  flags.DEFINE_integer(
       'inter_op_threads',
       default=0,
       help='inter op threads.')
  flags.DEFINE_integer(
       'intra_op_threads',
       default=0,
       help='intra op threads.')
  flags.DEFINE_integer(
       'validation_steps',
       default=None,
       help='Assign validation_steps steps when use train_and_eval.')
  flags.DEFINE_integer(
      'epochs',
      default=None,
      help='The number of epochs.')
  flags.DEFINE_integer(
      'start_epoch',
      default=None,
      help='The first epoch for finetuning.')
  flags.DEFINE_bool(
      'skip_eval',
      default=False,
      help='Whether to skip_eval per epoch')
  flags.DEFINE_string(
    'distribution_strategy',
    default=None,
    help="Use which distribution_strategy to train. eg. 'off', 'mlu', 'MirroredStrategy' etc." )
  flags.DEFINE_string(
    'finetune_checkpoint',
    default=None,
    help="Incoming the path of ckpt. " )

def serialize_config(params: base_configs.ExperimentConfig,
                     model_dir: str):
  """Serializes and saves the experiment config."""
  params_save_path = os.path.join(model_dir, 'params.yaml')
  logging.info('Saving experiment configuration to %s', params_save_path)
  tf.io.gfile.makedirs(model_dir)
  params_dict.save_params_dict_to_yaml(params, params_save_path)


def train_and_eval(
    params: base_configs.ExperimentConfig,
    strategy_override: tf.distribute.Strategy) -> Mapping[str, Any]:
  """Runs the train and eval path using compile/fit."""
  logging.info('Running train and eval.')

  if params.use_performance and params.use_profiler:
    raise ValueError("You can only set use_profiler or use_performance, not at the same time, otherwise the e2e time will be worse")

  if params.use_profiler:
    from tensorflow_core.python.client import timeline

  if params.use_horovod:
    import horovod.tensorflow.keras as hvd
    global hvd
    hvd.init()

  if params.use_performance:
    from record_time import TimeHistoryRecord, write_json
    global TimeHistoryRecord
    global write_json

  if params.use_qat:
    import tensorflow_model_optimization as tfmot
    global tfmot

  # Note: for TPUs, strategy and scope should be created before the dataset
  strategy = strategy_override or distribution_utils.get_distribution_strategy(
      distribution_strategy=params.runtime.distribution_strategy,
      all_reduce_alg=params.runtime.all_reduce_alg,
      num_gpus=params.runtime.num_gpus, num_mlus=params.runtime.num_mlus,
      tpu_address=params.runtime.tpu)

  strategy_scope = distribution_utils.get_strategy_scope(strategy)

  logging.info('Detected %d devices.',
               strategy.num_replicas_in_sync if strategy else 1)

  label_smoothing = params.model.loss.label_smoothing
  one_hot = label_smoothing and label_smoothing > 0

  builders = _get_dataset_builders(params, strategy, one_hot)

  if params.use_horovod:
    input_context = tf.distribute.InputContext(num_input_pipelines=hvd.size(), input_pipeline_id=hvd.rank(), num_replicas_in_sync=hvd.size())
  else:
    input_context = None

  datasets = []
  for builder in builders:
      if builder:
          if builder.is_training:
              datasets.append(builder.build(input_context))
          else:
              datasets.append(builder.build())
      else:
          datasets.append(None)

  # Unpack datasets and builders based on train/val/test splits
  train_builder, validation_builder = builders  # pylint: disable=unbalanced-tuple-unpacking
  train_dataset, validation_dataset = datasets

  # use the launch.sh to pass in steps and epochs parameters
  if flags.FLAGS.epochs != 0:
    train_epochs = flags.FLAGS.epochs
  else:
    train_epochs = params.train.epochs

  train_steps = params.train.steps or train_builder.num_steps
  if params.use_horovod and not params.train.steps:
    # If train_steps is set by train_builder.num_steps, note that
    # train_build.num_steps always return the step number of training 1 epoch using 1 card.
    # When use horovod, n cards finish 1 epoch by running train_steps // hvd.size() steps
    train_steps = train_steps // hvd.size()
  validation_steps = params.evaluation.steps or validation_builder.num_steps

  logging.info('Global batch size: %d', train_builder.global_batch_size)

  with strategy_scope:
    model_params = params.model.model_params.as_dict()
    model = get_models()[params.model.name](**model_params)
    learning_rate = optimizer_factory.build_learning_rate(
        params=params.model.learning_rate,
        batch_size=train_builder.global_batch_size
        if not params.use_horovod else train_builder.global_batch_size * hvd.size(),
        train_steps=train_steps)
    optimizer = optimizer_factory.build_optimizer(
        optimizer_name=params.model.optimizer.name,
        base_learning_rate=learning_rate,
        params=params.model.optimizer.as_dict())

    if params.use_horovod:
      with tf.keras.utils.CustomObjectScope({learning_rate.__class__.__name__: learning_rate}):
        optimizer = hvd.DistributedOptimizer(optimizer)

    if params.use_amp:
      optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    initialize(params, train_builder)

    metrics_map = _get_metrics(one_hot)
    metrics = [metrics_map[metric] for metric in params.train.metrics]

    if one_hot:
      loss_obj = tf.keras.losses.CategoricalCrossentropy(
          label_smoothing=params.model.loss.label_smoothing,
          reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE)
    else:
      loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()

    run_metadata = None
    run_options = None
    if params.use_profiler and (not params.use_horovod or (params.use_horovod and hvd.rank() == 0)):
      run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
      run_metadata = tf.compat.v1.RunMetadata()

    model.compile(optimizer=optimizer,
                  loss=loss_obj,
                  metrics=metrics,
                  options=run_options,
                  run_metadata=run_metadata)

    initial_epoch = 0
    if params.train.resume_checkpoint:
      initial_epoch = resume_from_checkpoint(model=model,
                                             model_dir=params.model_dir,
                                             train_steps=train_steps,
                                             finetune_checkpoint=params.finetune_checkpoint,
                                             params=params)
  # quantization aware training with tfmot
  if params.use_qat:
    tfmot_quantize_model = tfmot.quantization.keras.quantize_model
    # quantize origin model
    model = tfmot_quantize_model(model)
    model.compile(optimizer=optimizer,
                  loss=loss_obj,
                  metrics=metrics,
                  options=run_options,
                  run_metadata=run_metadata)

  serialize_config(params=params, model_dir=params.model_dir)
  # TODO(dankondratyuk): callbacks significantly slow down training

  file_log = open(params.model_dir + '/loss.log', 'a')
  print_loss_hook = tf.keras.callbacks.LambdaCallback(
      on_epoch_begin=lambda epoch, logs: file_log.write("\nstart epoch: %d\n" % epoch),
      on_batch_end=lambda batch, logs: file_log.write("step: %d, loss: %.6f\n" % (batch, logs['loss'])))

  callbacks = []
  if ((params.use_horovod == True) and (hvd.rank() == 0)) or (params.use_horovod == False):
    callbacks = custom_callbacks.get_callbacks(
        model_checkpoint=params.train.callbacks.enable_checkpoint_and_export,
        include_tensorboard=False,
        time_history=False,
        track_lr=params.train.tensorboard.track_lr,
        write_model_weights=params.train.tensorboard.write_model_weights,
        initial_step=initial_epoch * train_steps,
        batch_size=train_builder.global_batch_size,
        log_steps=params.train.time_history.log_steps,
        model_dir=params.model_dir)
    callbacks.append(print_loss_hook)
  if params.use_horovod == True:
    callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
  if params.use_performance and (not params.use_horovod or (params.use_horovod and hvd.rank() == 0)):
    time_callback = TimeHistoryRecord()
    callbacks.append(time_callback)


  if flags.FLAGS.skip_eval:
  # if params.evaluation.skip_eval:
    validation_kwargs = {}
  else:
    validation_kwargs = {
        'validation_data': validation_dataset,
        'validation_steps': validation_steps,
        'validation_freq': params.evaluation.epochs_between_evals,
    }

  verbose = 1 if ((params.use_horovod and hvd.rank() == 0) or not params.use_horovod) else 0

  history = model.fit(
      train_dataset,
      epochs=train_epochs,
      steps_per_epoch=params.finetune_steps or train_steps,
      initial_epoch=initial_epoch,
      callbacks=callbacks,
      verbose=verbose,
      **validation_kwargs)

  validation_output = None
  if not params.evaluation.skip_eval:
    if (params.use_horovod and hvd.rank()==0) or (not params.use_horovod):
      validation_output = model.evaluate(
          validation_dataset, steps=validation_steps, verbose=2)

  # TODO(dankondratyuk): eval and save final test accuracy
  stats = common.build_stats(history,
                             validation_output,
                             callbacks)

  if params.use_profiler and (not params.use_horovod or (params.use_horovod and hvd.rank() == 0)):
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    if not tf.io.gfile.exists("profiler"):
        tf.io.gfile.mkdir("profiler")
    with open("profiler/timeline.json", 'w') as f:
        f.write(ctf)
  if params.use_performance:
    global_batch_size = train_builder.global_batch_size if not params.use_horovod else train_builder.global_batch_size * hvd.size()
    if not params.use_horovod or (params.use_horovod and hvd.rank() == 0):
      write_json("summary", global_batch_size, time_callback.times)
  return stats

def eval(
    params: base_configs.ExperimentConfig,
    strategy_override: tf.distribute.Strategy) -> Mapping[str, Any]:
  """Runs the eval path using compile/fit."""
  logging.info('Running eval.')


  if params.use_horovod:
    import horovod.tensorflow.keras as hvd
    global hvd
    hvd.init()

  if params.use_qat:
    import tensorflow_model_optimization as tfmot
    global tfmot

  # Note: for TPUs, strategy and scope should be created before the dataset
  strategy = strategy_override or distribution_utils.get_distribution_strategy(
      distribution_strategy=params.runtime.distribution_strategy,
      all_reduce_alg=params.runtime.all_reduce_alg,
      num_gpus=params.runtime.num_gpus, num_mlus=params.runtime.num_mlus,
      tpu_address=params.runtime.tpu)

  strategy_scope = distribution_utils.get_strategy_scope(strategy)

  logging.info('Detected %d devices.',
               strategy.num_replicas_in_sync if strategy else 1)

  label_smoothing = params.model.loss.label_smoothing
  one_hot = label_smoothing and label_smoothing > 0

  builders = _get_dataset_builders(params, strategy, one_hot)
  datasets = [builder.build() if builder else None for builder in builders]

  # Unpack datasets and builders based on train/val/test splits
  train_builder, validation_builder = builders  # pylint: disable=unbalanced-tuple-unpacking
  train_dataset, validation_dataset = datasets

  train_epochs = params.train.epochs
  train_steps = params.train.steps or train_builder.num_steps
  validation_steps = params.evaluation.steps or validation_builder.num_steps

  initialize(params, train_builder)

  logging.info('Global batch size: %d', train_builder.global_batch_size)

  with strategy_scope:
    model_params = params.model.model_params.as_dict()
    model = get_models()[params.model.name](**model_params)
    learning_rate = optimizer_factory.build_learning_rate(
        params=params.model.learning_rate,
        batch_size=train_builder.global_batch_size,
        train_steps=train_steps)
    optimizer = optimizer_factory.build_optimizer(
        optimizer_name=params.model.optimizer.name,
        base_learning_rate=learning_rate,
        params=params.model.optimizer.as_dict())

    if params.use_horovod:
       with tf.keras.utils.CustomObjectScope({learning_rate.__class__.__name__: learning_rate}):
         optimizer = hvd.DistributedOptimizer(optimizer)

    metrics_map = _get_metrics(one_hot)
    metrics = [metrics_map[metric] for metric in params.train.metrics]

    if one_hot:
      loss_obj = tf.keras.losses.CategoricalCrossentropy(
          label_smoothing=params.model.loss.label_smoothing,
          reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE)
          #reduction=tf.keras.losses.Reduction.NONE)
    else:
      loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer,
                  loss=loss_obj,
                  metrics=metrics)

    if params.use_qat:
      tfmot_quantize_model = tfmot.quantization.keras.quantize_model
      # quantize origin model
      model = tfmot_quantize_model(model)
      model.compile(optimizer=optimizer,
                    loss=loss_obj,
                    metrics=metrics)

    initial_epoch = 0
    if params.train.resume_checkpoint:
      initial_epoch = resume_from_checkpoint(model=model,
                                             model_dir=params.model_dir,
                                             train_steps=train_steps,
                                             finetune_checkpoint=params.finetune_checkpoint,
                                             params=params)

  serialize_config(params=params, model_dir=params.model_dir)

  validation_kwargs = {
      'validation_data': validation_dataset,
      'validation_steps': validation_steps,
      'validation_freq': params.evaluation.epochs_between_evals,
  }

  validation_output = model.evaluate(
      validation_dataset, steps=validation_steps, verbose=1)
  print('Run stats:\ntop_1 acc: %s\t eval_loss: %s'%(validation_output[1], validation_output[0]))
  with open(params.model_dir + "/" + "Run_stats_eval_%s.txt"%(params.model.name), 'w') as fout:
    fout.write('Run stats:\n%s\t%s'%(validation_output[1], validation_output[0]))

def export(params: base_configs.ExperimentConfig):
  """Runs the model export functionality."""
  logging.info('Exporting model.')
  model_params = params.model.model_params.as_dict()
  model = get_models()[params.model.name](**model_params)
  checkpoint = params.export.checkpoint
  if checkpoint is None:
    logging.info('No export checkpoint was provided. Using the latest '
                 'checkpoint from model_dir.')
    checkpoint = tf.train.latest_checkpoint(params.model_dir)

  model.load_weights(checkpoint)
  model.save(params.export.destination)


def run(flags_obj: flags.FlagValues,
        strategy_override: tf.distribute.Strategy = None) -> Mapping[str, Any]:
  """Runs Image Classification model using native Keras APIs.

  Args:
    flags_obj: An object containing parsed flag values.
    strategy_override: A `tf.distribute.Strategy` object to use for model.

  Returns:
    Dictionary of training/eval stats
  """
  params = _get_params_from_flags(flags_obj)
  if params.mode == 'train_and_eval':
    return train_and_eval(params, strategy_override)
  elif params.mode == 'export_only':
    export(params)
  elif params.mode == 'eval':
    eval(params, strategy_override)
  else:
    raise ValueError('{} is not a valid mode.'.format(params.mode))


def main(_):
  flags.FLAGS.config_file = "models/vision/configs/" + flags.FLAGS.model_type + ".yaml"
  with logger.benchmark_context(flags.FLAGS):
    stats = run(flags.FLAGS)
  if stats and ((flags.FLAGS.use_horovod and hvd.rank() == 0) or not flags.FLAGS.use_horovod):
    logging.info('Run stats:\n%s', stats)
    with open("Run_stats.txt", 'w') as fout:
       fout.write('Run stats:\n%s'%stats)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  define_classifier_flags()
  flags.mark_flag_as_required('data_dir')
  flags.mark_flag_as_required('mode')
  flags.mark_flag_as_required('model_type')
  flags.mark_flag_as_required('dataset')

  app.run(main)
