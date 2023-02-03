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

import os
import json
import pprint
from typing import Any, Tuple, Text, Optional, Mapping

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from models.common import distribute_utils
from models.modeling import hyperparams
from models.modeling import performance
from models.utils import hyperparams_flags
from models.utils.misc import keras_utils
from models.vision import callbacks as custom_callbacks
from models.vision import dataset_factory
from models.vision import optimizer_factory
from models.vision.configs import base_configs
from models.vision.configs import configs
from models.vision.efficientnet import efficientnet_model
from models.vision.resnet import common
from models.vision.resnet import resnet_model
from models.vision.inceptionv2 import inceptionv2_model
from models.vision.inceptionv3 import inceptionv3_model
from models.vision.vgg import vgg_model
from models.vision.densenet import densenet
from models.vision.resnet import resnet18_model
from models.vision.resnet import resnet101_model


def get_models() -> Mapping[str, tf.keras.Model]:
  """Returns the mapping from models.type name to Keras model."""
  return {
      'efficientnet': efficientnet_model.EfficientNet.from_name,
      'resnet50': resnet_model.resnet50,
      'densenet201': densenet.DenseNet201,
      'vgg16': vgg_model.vgg16,
      'vgg19': vgg_model.vgg19,
      'resnet101': resnet101_model.resnet101,
      'resnet18': resnet18_model.resnet18,
      'inceptionv2': inceptionv2_model.inceptionv2,
      'inceptionv3': inceptionv3_model.inceptionv3,
  }


def get_dtype_map() -> Mapping[str, tf.dtypes.DType]:
  """Returns the mapping from dtype string representations to TF dtypes."""
  return {
      'float32': tf.float32,
      'bfloat16': tf.bfloat16,
      'float16': tf.float16,
      'fp32': tf.float32,
      'bf16': tf.bfloat16,
  }


def _get_metrics(one_hot: bool) -> Mapping[Text, Any]:
  """Get a dict of available metrics to track."""
  if one_hot:
    return {
        # (name, metric_fn)
        'acc':
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        'accuracy':
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        'top_1':
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        'top_5':
            tf.keras.metrics.TopKCategoricalAccuracy(
                k=5, name='top_5_accuracy'),
    }
  else:
    return {
        # (name, metric_fn)
        'acc':
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        'accuracy':
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        'top_1':
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        'top_5':
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=5, name='top_5_accuracy'),
    }


def get_image_size_from_model(
    params: base_configs.ExperimentConfig) -> Optional[int]:
  """If the given model has a preferred image size, return it."""
  if params.model_name == 'efficientnet':
    efficientnet_name = params.model.model_params.model_name
    if efficientnet_name in efficientnet_model.MODEL_CONFIGS:
      return efficientnet_model.MODEL_CONFIGS[efficientnet_name].resolution
  return None


def _get_dataset_builders(params: base_configs.ExperimentConfig,
                          strategy: tf.distribute.Strategy,
                          one_hot: bool) -> Tuple[Any, Any]:
  """Create and return train and validation dataset builders."""
  if one_hot:
    logging.warning('label_smoothing > 0, so datasets will be one hot encoded.')
  else:
    logging.warning('label_smoothing not applied, so datasets will not be one '
                    'hot encoded.')

  num_devices = strategy.num_replicas_in_sync if strategy else 1

  image_size = get_image_size_from_model(params)

  dataset_configs = [params.train_dataset, params.validation_dataset]
  builders = []

  for config in dataset_configs:
    if config is not None and config.has_data:
      builder = dataset_factory.DatasetBuilder(
          config,
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
  elif (params.train_dataset.dtype == 'float32' or
        params.train_dataset.dtype == 'bfloat16'):
    return 1.
  else:
    assert params.train_dataset.dtype == 'float16'
    return 'dynamic'


def _get_params_from_flags(flags_obj: flags.FlagValues):
  """Get ParamsDict from flags."""
  model = flags_obj.model_type.lower()
  dataset = flags_obj.dataset.lower()
  params = configs.get_config(model=model, dataset=dataset)

  dtype = 'float16' if flags_obj.use_amp else 'float32'

  flags_overrides = {
      'model_dir': flags_obj.model_dir,
      'mode': flags_obj.mode,
      'model': {
          'name': model,
      },
      'runtime': {
          'run_eagerly': flags_obj.run_eagerly,
          'tpu': flags_obj.tpu,
          'num_mlus': flags_obj.num_mlus,
          'num_gpus': flags_obj.num_gpus,
          'distribution_strategy': flags_obj.distribution_strategy,
      },
      'train_dataset': {
          'data_dir': flags_obj.data_dir,
          'dtype': dtype,
      },
      'validation_dataset': {
          'data_dir': flags_obj.data_dir,
      },
      'train': {
          'time_history': {
              'log_steps': flags_obj.log_steps,
          },
          'callbacks': {
              'enable_tensorboard': flags_obj.enable_tensorboard,
          },
      },
      'evaluation': {
          'skip_eval': flags_obj.skip_eval,
      },
      'finetune_steps': flags_obj.finetune_steps,
      'finetune_checkpoint' : flags_obj.finetune_checkpoint,
      'use_horovod': flags_obj.use_horovod,
      'use_performance': flags_obj.use_performance,
      'datasets_num_private_threads': flags_obj.datasets_num_private_threads,
  }

  if flags_obj.batch_size != 0:
    flags_overrides['train_dataset']['batch_size'] = flags_obj.batch_size
  if flags_obj.epochs != 0:
    flags_overrides['train']['epochs'] = flags_obj.epochs

  overriding_configs = (flags_obj.config_file, flags_obj.params_override,
                        flags_overrides)

  pp = pprint.PrettyPrinter()

  logging.info('Base params: %s', pp.pformat(params.as_dict()))

  for param in overriding_configs:
    logging.info('Overriding params: %s', param)
    params = hyperparams.override_params_dict(params, param, is_strict=False)

  params.validate()
  params.lock()

  logging.info('Final model parameters: %s', pp.pformat(params.as_dict()))
  return params


def resume_from_checkpoint(model: tf.keras.Model, model_dir: str,
                           train_steps: int, finetune_checkpoint: str) -> int:
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
  initial_epoch = model.optimizer.iterations // train_steps
  logging.info('Completed loading from checkpoint.')
  logging.info('Resuming from epoch %d', initial_epoch)
  return int(initial_epoch)


def initialize(params: base_configs.ExperimentConfig,
               dataset_builder: dataset_factory.DatasetBuilder):
  """Initializes backend related initializations."""
  keras_utils.set_session_config(enable_xla=params.runtime.enable_xla)
  performance.set_mixed_precision_policy(dataset_builder.dtype,
                                         get_loss_scale(params))
  if tf.config.list_physical_devices('GPU') and params.model.model_name not in ['densenet201', 'inceptionv2', 'inceptionv3']:
    data_format = 'channels_first'
  else:
    data_format = 'channels_last'
  tf.keras.backend.set_image_data_format(data_format)
  if params.runtime.run_eagerly:
    # Enable eager execution to allow step-by-step debugging
    tf.config.experimental_run_functions_eagerly(True)
  if tf.config.list_physical_devices('GPU'):
    if params.runtime.gpu_thread_mode:
      keras_utils.set_gpu_thread_mode_and_count(
          per_gpu_thread_count=params.runtime.per_gpu_thread_count,
          gpu_thread_mode=params.runtime.gpu_thread_mode,
          num_gpus=params.runtime.num_gpus,
          datasets_num_private_threads=params.runtime
          .dataset_num_private_threads)  # pylint:disable=line-too-long
    if params.runtime.batchnorm_spatial_persistent:
      os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'


def define_classifier_flags():
  """Defines common flags for image classification."""
  hyperparams_flags.initialize_common_flags()
  flags.DEFINE_string(
      'data_dir', default=None, help='The location of the input data.')
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
  flags.DEFINE_bool(
       'use_horovod',
       default=False,
       help='Use hvd to train networks.')
  flags.DEFINE_string(
      'dataset',
      default=None,
      help='The name of the dataset, e.g. ImageNet, etc.')
  flags.DEFINE_integer(
      'log_steps',
      default=100,
      help='The interval of steps between logging of batch level stats.')
  flags.DEFINE_integer(
      'batch_size',
      default=None,
      help='The training batch size.(single card)')
  flags.DEFINE_integer(
      'epochs',
      default=None,
      help='The total training epochs')
  flags.DEFINE_bool(
      'skip_eval',
      default=False,
      help='Whether to skip_eval per epoch')
  flags.DEFINE_integer(
      'finetune_steps',
      default=None,
      help='Assign finetune steps when run less than 1 epoch.')
  flags.DEFINE_integer(
      'num_mlus',
      default=None,
      help='The number of mlu devices.')
  flags.DEFINE_string(
      'distribution_strategy',
      default=None,
      help="Use which distribution_strategy to train. eg. 'off', 'one_device', 'mirrored' etc."
      )
  flags.DEFINE_string(
    'finetune_checkpoint',
    default=None,
    help="Incoming the path of ckpt. " )
  flags.DEFINE_bool(
      'enable_tensorboard',
      default=False,
      help='If use tensorboard, please set True.')
  flags.DEFINE_bool(
      'use_performance',
      default=False,
      help='If use performance test tools, please set True.')
  flags.DEFINE_bool(
      'use_amp',
      default=False,
      help='If use amp, please set True.')
  flags.DEFINE_integer(
      'datasets_num_private_threads',
      default=None,
      help='Number of threads for a private threadpool.')

def serialize_config(params: base_configs.ExperimentConfig, model_dir: str):
  """Serializes and saves the experiment config."""
  params_save_path = os.path.join(model_dir, 'params.yaml')
  logging.info('Saving experiment configuration to %s', params_save_path)
  tf.io.gfile.makedirs(model_dir)
  hyperparams.save_params_dict_to_yaml(params, params_save_path)

def train_and_eval(
    params: base_configs.ExperimentConfig,
    strategy_override: tf.distribute.Strategy) -> Mapping[str, Any]:
  """Runs the train and eval path using compile/fit."""
  logging.info('Running train and eval.')

  if params.use_horovod:
    import horovod.tensorflow.keras as hvd
    global hvd
    hvd.init()

  if params.use_performance:
      from record_time import TimeHistoryRecord, write_json
      global TimeHistoryRecord
      global write_json

  if params.use_horovod:
    if flags.FLAGS.num_mlus > 0 and flags.FLAGS.num_gpus == 0:
      mlus = tf.config.experimental.list_physical_devices('MLU')
      if mlus:
        tf.config.experimental.set_visible_devices(mlus[hvd.local_rank()], 'MLU')
    elif flags.FLAGS.num_mlus == 0 and flags.FLAGS.num_gpus > 0:
      gpus = tf.config.experimental.list_physical_devices('GPU')
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    else:
      raise ValueError("Horovod can only be used when only one of gpu or mlu is greater than 0")

  distribute_utils.configure_cluster(params.runtime.worker_hosts,
                                     params.runtime.task_index)

  # Note: for TPUs, strategy and scope should be created before the dataset
  # TODO(chencongying): Relevant content needs to be modified after TF2 supports distributed
  strategy = strategy_override or distribute_utils.get_distribution_strategy(
      distribution_strategy=params.runtime.distribution_strategy,
      all_reduce_alg=params.runtime.all_reduce_alg,
      num_gpus=params.runtime.num_gpus,
      num_mlus=params.runtime.num_mlus,
      tpu_address=params.runtime.tpu)

  strategy_scope = distribute_utils.get_strategy_scope(strategy)

  logging.info('Detected %d devices.',
               strategy.num_replicas_in_sync if strategy else 1)

  label_smoothing = params.model.loss.label_smoothing
  one_hot = label_smoothing and label_smoothing > 0

  builders = _get_dataset_builders(params, strategy, one_hot)
  datasets = [
      builder.build(strategy) if builder else None for builder in builders
  ]

  # Unpack datasets and builders based on train/val/test splits
  train_builder, validation_builder = builders  # pylint: disable=unbalanced-tuple-unpacking
  train_dataset, validation_dataset = datasets

  if params.datasets_num_private_threads:
    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = (
        params.datasets_num_private_threads)
    train_dataset = train_dataset.with_options(options)
    logging.info('datasets_num_private_threads: %s',
                 params.datasets_num_private_threads)

  train_epochs = params.train.epochs
  train_steps = params.train.steps or train_builder.num_steps

  if params.use_horovod and not params.train.steps:
    # If train_steps is set by train_builder.num_steps, note that
    # train_build.num_steps always return the step number of training 1 epoch using 1 card.
    # When use horovod, n cards finish 1 epoch by running train_steps // hvd.size() steps
    train_steps = train_steps // hvd.size()

  validation_steps = params.evaluation.steps or validation_builder.num_steps

  initialize(params, train_builder)

  logging.info('Global batch size: %d', train_builder.global_batch_size)

  with strategy_scope:
    model_params = params.model.model_params.as_dict()
    model = get_models()[params.model.name](**model_params)
    learning_rate = optimizer_factory.build_learning_rate(
        params=params.model.learning_rate,
        batch_size=train_builder.global_batch_size
        if not params.use_horovod else train_builder.global_batch_size * hvd.size(),
        train_epochs=train_epochs,
        train_steps=train_steps)
    optimizer = optimizer_factory.build_optimizer(
        optimizer_name=params.model.optimizer.name,
        base_learning_rate=learning_rate,
        params=params.model.optimizer.as_dict(),
        model=model)

    if params.use_horovod:
      with tf.keras.utils.CustomObjectScope({learning_rate.__class__.__name__: learning_rate}):
        optimizer = hvd.DistributedOptimizer(optimizer)

    metrics_map = _get_metrics(one_hot)
    metrics = [metrics_map[metric] for metric in params.train.metrics]
    steps_per_loop = train_steps if params.train.set_epoch_loop else 1

    if one_hot:
      loss_obj = tf.keras.losses.CategoricalCrossentropy(
          label_smoothing=params.model.loss.label_smoothing)
    else:
      loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(
        optimizer=optimizer,
        loss=loss_obj,
        metrics=metrics,
        experimental_steps_per_execution=steps_per_loop)

    initial_epoch = 0
    if params.train.resume_checkpoint:
      initial_epoch = resume_from_checkpoint(
          model=model, model_dir=params.model_dir, train_steps=train_steps,
          finetune_checkpoint=params.finetune_checkpoint)

    callbacks = []
    if (params.use_horovod and (hvd.rank() == 0)) or not params.use_horovod:
      callbacks = custom_callbacks.get_callbacks(
          model_checkpoint=params.train.callbacks.enable_checkpoint_and_export,
          include_tensorboard=params.train.callbacks.enable_tensorboard,
          profile_batch=(2,2),
          time_history=params.train.callbacks.enable_time_history,
          track_lr=params.train.tensorboard.track_lr,
          write_model_weights=params.train.tensorboard.write_model_weights,
          initial_step=initial_epoch * train_steps,
          batch_size=train_builder.global_batch_size,
          log_steps=params.train.time_history.log_steps,
          model_dir=params.model_dir,
          backup_and_restore=params.train.callbacks.enable_backup_and_restore)
    if params.use_horovod:
      callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))

  serialize_config(params=params, model_dir=params.model_dir)

  if params.evaluation.skip_eval:
    validation_kwargs = {}
  else:
    validation_kwargs = {
        'validation_data': validation_dataset,
        'validation_steps': validation_steps,
        'validation_freq': params.evaluation.epochs_between_evals,
    }

  if params.use_performance and (not params.use_horovod or (params.use_horovod and hvd.rank() == 0)):
    time_callback = TimeHistoryRecord()
    callbacks.append(time_callback)
  history = model.fit(
      train_dataset,
      epochs=train_epochs,
      steps_per_epoch=params.finetune_steps if params.finetune_steps != 0 else train_steps,
      initial_epoch=initial_epoch,
      callbacks=callbacks,
      verbose=1 if ((params.use_horovod and hvd.rank() == 0) or not params.use_horovod) else 0,
      **validation_kwargs)
  if params.use_performance and (not params.use_horovod or (params.use_horovod and hvd.rank() == 0)):
    global_batch_size = train_builder.global_batch_size if not params.use_horovod else train_builder.global_batch_size * hvd.size()
    write_json("summary", global_batch_size, time_callback.times)

  validation_output = None
  if not params.evaluation.skip_eval:
    if (params.use_horovod and hvd.rank()==0) or (not params.use_horovod):
      validation_output = model.evaluate(
          validation_dataset, steps=validation_steps, verbose=2)

  # TODO(dankondratyuk): eval and save final test accuracy
  stats = common.build_stats(history, validation_output, callbacks)
  return stats


def export(params: base_configs.ExperimentConfig):
  """Runs the model export functionality."""
  logging.info('Exporting model.')
  model_params = params.model.model_params.as_dict()
  model = get_models()[params.model.name](**model_params)
  checkpoint = params.export.checkpoint
  if checkpoint is None:
    logging.info('No export checkpoint was provided. Using the latest '
                 'checkpoint from models.dir.')
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

  # Contains the ip addresses and corresponding ports of multiple devices,
  # which are used for multiple servers to communicate using grpc.
  # for example:
  # 'worker': ["xx.xx.xx.xx:12345", "zz.zz.zz.zz:23456"]
  if flags_obj.distribution_strategy == "multi_worker_mirrored":
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': ["127.0.0.1:12345", "127.0.0.1:23456"]
        },
        'task': {'type': 'worker', 'index': 0}
    })

  if params.mode == 'train_and_eval':
    return train_and_eval(params, strategy_override)
  elif params.mode == 'export_only':
    export(params)
  else:
    raise ValueError('{} is not a valid mode.'.format(params.mode))


def main(_):
  flags.FLAGS.config_file = "models/vision/configs/" + flags.FLAGS.model_type + ".yaml"
  stats = run(flags.FLAGS)
  if stats and ((flags.FLAGS.use_horovod and hvd.rank() == 0) or not flags.FLAGS.use_horovod):
    logging.info('Run stats:\n%s', stats)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  define_classifier_flags()
  flags.mark_flag_as_required('data_dir')
  flags.mark_flag_as_required('mode')
  flags.mark_flag_as_required('model_type')
  flags.mark_flag_as_required('dataset')

  app.run(main)

