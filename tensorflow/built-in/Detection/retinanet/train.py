# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Main function to train various object detection models."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import functools
import os
import pprint
import numpy as np
import tensorflow as tf

from models.modeling.hyperparams import params_dict
#from modeling.training import distributed_executor as executor
from models.utils import hyperparams_flags
from models.configs import factory as config_factory
from models.dataloader import input_reader
from models.dataloader import mode_keys as ModeKeys
from models.modeling import factory as model_factory
from models import distribution_utils

from models.modeling.learning_rates import StepLearningRateWithLinearWarmup

hyperparams_flags.initialize_common_flags()

flags.DEFINE_integer(
    'num_mlus',
    default=1,
    help='How many MLUs to use with the DistributionStrategies API. The '
         'default is 1.')

flags.DEFINE_bool(
    'enable_xla',
    default=False,
    help='Enable XLA for GPU')

flags.DEFINE_string(
    'mode',
    default='train',
    help='Mode to run: `train`, `eval` or `train_and_eval`.')

flags.DEFINE_string(
    'model', default='retinanet',
    help='Model to run: `retinanet` or `shapemask`.')

flags.DEFINE_string('training_file_pattern', None,
                    'Location of the train data.')

flags.DEFINE_string('eval_file_pattern', None, 'Location of the eval data')

flags.DEFINE_string('val_json_file', None, 'Instances json file of the eval data')

flags.DEFINE_string(
    'checkpoint_path', None, 'The pretrained model.')

flags.DEFINE_integer(
    'batch_size',
    default=8,
    help="batch size for training.")

flags.DEFINE_integer(
    'total_steps',
    default=1833*1*50,
    help="How many steps to train. default is 1833*1*50.")

flags.DEFINE_integer(
    'iterations_per_loop',
    default=1833*1,
    help="How many steps to make in each estimator call. default is 1833*1.")

flags.DEFINE_float(
    'init_learning_rate',
    default=0.01,
    help="Initial value of learning rate. default is 0.01")

flags.DEFINE_string(
    'learning_rate_levels',
    "[0.001, 0.0001]",
    "the range of learning rate.")

flags.DEFINE_string(
    'learning_rate_steps',
    "[1833*8*30, 1833*8*40]",
    "the step interval of learning rate.")

flags.DEFINE_bool("use_horovod", False, "Whether to use horovod for multiple devices runs")

flags.DEFINE_bool(
    'use_profiler',
    default=False,
    help='Whether to use performance test tools')

flags.DEFINE_bool(
    'use_performance',
    default=False,
    help='Whether to use performance test tools')

flags.DEFINE_bool("use_amp", False, "If use amp, please set True")

FLAGS = flags.FLAGS

def build_model_fn(features, labels, mode, params):

    model_builder = model_factory.model_generator(params)
    #model = model_builder.build_model(params, mode=mode)

    loss_fn = model_builder.build_loss_fn()
    global_step = tf.train.get_or_create_global_step()
    #outputs = model(features, training=True)
    outputs = model_builder.build_outputs(features, mode)
    prediction_loss = loss_fn(labels, outputs)

    trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    filter_fn = model_builder.make_filter_trainable_variables_fn()
    no_frozen_variables = filter_fn(trainable_variables)

    l2_loss = model_builder.weight_decay_loss(no_frozen_variables)
    detection_loss = tf.reduce_mean(prediction_loss['total_loss'])
    if (FLAGS.num_mlus > 0) and (FLAGS.num_gpus == 0):
      total_loss = (detection_loss + l2_loss) / FLAGS.num_mlus
    elif (FLAGS.num_gpus > 0) and (FLAGS.num_mlus == 0):
      total_loss = (detection_loss + l2_loss) / FLAGS.num_gpus
    elif (FLAGS.num_gpus == 0) and (FLAGS.num_mlu == 0):
      total_loss = detection_loss + l2_loss
    else:
      raise ValueError('Both mlu and gpu cannot be less than 0, and cannot both be greater than 0.')
    lr_cfg = params.train.learning_rate
    lr_builder = StepLearningRateWithLinearWarmup(lr_cfg)
    lr = lr_builder(global_step)

    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)

    if FLAGS.use_horovod:
      optimizer = hvd.DistributedOptimizer(optimizer)

    if FLAGS.use_amp:
      optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    tf.summary.scalar("total_loss", total_loss)
    tf.summary.scalar("learning_rate", lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(total_loss, global_step)
        return tf.estimator.EstimatorSpec(
                                  mode=mode,
                                  loss=total_loss,
                                  train_op=train_op,
                                  )

def _ensure_dir(log_dir):
  """Makes log dir if not existed."""
  if not tf.io.gfile.exists(log_dir):
    tf.io.gfile.makedirs(log_dir)

def run(callbacks=None):
  _ensure_dir(FLAGS.model_dir)

  params = config_factory.config_generator(FLAGS.model)

  params = params_dict.override_params_dict(
      params, FLAGS.params_override, is_strict=True)
  FLAGS.learning_rate_levels=eval(FLAGS.learning_rate_levels)
  FLAGS.learning_rate_steps=eval(FLAGS.learning_rate_steps)

  params.override(
      {
          'strategy_type': FLAGS.strategy_type,
          'model_dir': FLAGS.model_dir,
          'train': {
            'batch_size': FLAGS.batch_size,
            'total_steps': FLAGS.total_steps,
            'input_sharding': FLAGS.use_horovod,
            'iterations_per_loop': FLAGS.iterations_per_loop,
            'learning_rate': {
               'init_learning_rate': FLAGS.init_learning_rate,
               'learning_rate_levels': FLAGS.learning_rate_levels,
               'learning_rate_steps': FLAGS.learning_rate_steps,
            },
            'train_file_pattern': FLAGS.training_file_pattern,
          },
          'eval' : {
            'eval_file_pattern': FLAGS.eval_file_pattern,
            'val_json_file': FLAGS.val_json_file,
          }
      },
      is_strict=False)

  params.validate()
  params.lock()
  pp = pprint.PrettyPrinter()
  params_str = pp.pformat(params.as_dict())
  logging.info('Model Parameters: {}'.format(params_str))

  train_input_fn = None
  training_file_pattern = FLAGS.training_file_pattern or params.train.train_file_pattern
  if not training_file_pattern and not FLAGS.eval_file_pattern:
    raise ValueError('Must provide at least one of training_file_pattern and '
                     'eval_file_pattern.')

  if FLAGS.use_performance and FLAGS.use_profiler:
    raise ValueError("You can only set use_profiler or use_performance, not at the same time, otherwise the e2e time will be worse")

  training_hooks = []
  master_process = True

  if training_file_pattern:
    if FLAGS.use_horovod:
      ctx_ = tf.distribute.InputContext(num_input_pipelines=hvd.size(),input_pipeline_id=hvd.rank(),num_replicas_in_sync=1)
      train_input_fn = input_reader.InputFn(
          file_pattern=training_file_pattern,
          params=params,
          mode=input_reader.ModeKeys.TRAIN,
          batch_size=params.train.batch_size,
          ctx=ctx_)
    else:
      train_input_fn = input_reader.InputFn(
          file_pattern=training_file_pattern,
          params=params,
          mode=input_reader.ModeKeys.TRAIN,
          batch_size=params.train.batch_size)

  distribution_strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=FLAGS.strategy_type,
      num_gpus=FLAGS.num_gpus)

  global_batch_size = params.train.batch_size
  if FLAGS.use_horovod:
    global_batch_size = params.train.batch_size * hvd.size()
  config = tf.ConfigProto()
  config.allow_soft_placement=True

  if FLAGS.use_horovod:
    tf.compat.v1.logging.info("hvd.size() = %d hvd.rank() = %d", hvd.size(), hvd.rank())
    master_process = (hvd.rank() == 0)
    if FLAGS.num_gpus > 0:
      config.gpu_options.visible_device_list = str(hvd.local_rank())
    elif FLAGS.num_mlus > 0:
      config.mlu_options.visible_device_list = str(hvd.local_rank())
    training_hooks.append(hvd.BroadcastGlobalVariablesHook(0))

  if FLAGS.use_profiler and master_process:
    timeline_hook = tf.estimator.ProfilerHook(save_steps=5, output_dir='profiler/')
    training_hooks.append(timeline_hook)

  run_config = tf.estimator.RunConfig(
          session_config=config,
          train_distribute=distribution_strategy,
          log_step_count_steps=1 if master_process else None,
          save_checkpoints_steps=10000 if master_process else None,
          keep_checkpoint_max=15)

  ssd_detector = tf.estimator.Estimator(
        model_fn=build_model_fn,
        model_dir=FLAGS.model_dir,
        params=params,
        config=run_config,
        warm_start_from=FLAGS.checkpoint_path,
        )

  if FLAGS.use_performance and master_process:
    time_hooks = TimeHook()
    training_hooks.append(time_hooks)


  ssd_detector.train(input_fn=train_input_fn,
                     max_steps=params.train.total_steps, hooks=training_hooks)

  if FLAGS.use_performance and master_process:
      write_json("summary", global_batch_size, time_hooks.times)

def main(argv):
  del argv  # Unused.

  if FLAGS.use_horovod:
    if FLAGS.num_gpus != 1 and FLAGS.num_mlus != 1:
      raise ValueError('Must make use_mlus=1 or use_gpus=1 when use horovod.')

    import horovod.tensorflow as hvd
    global hvd
    hvd.init()

  if FLAGS.use_performance:
      from record_time import TimeHook, write_json
      global TimeHook
      global write_json

  run()


if __name__ == '__main__':

  app.run(main)
