# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
"""Binary to run train and evaluation on object detection model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow as tf
import dllogger
import time
import os

from models.research.object_detection import model_hparams
from models.research.object_detection import model_lib
from models.research.object_detection.utils.exp_utils import AverageMeter, setup_dllogger

import dllogger

from models.research.object_detection.utils import config_util

flags.DEFINE_boolean('use_horovod', False,
                     'Whether use horovod to do allreduce.')
flags.DEFINE_boolean('use_profiler', False,
                     'Whether use profiler.')
flags.DEFINE_boolean('use_performance', False,
                     'Whether use performance test tools.')
flags.DEFINE_string(
    'model_dir', None, 'Path to output model directory '
    'where event and checkpoint files will be written.')
flags.DEFINE_string(
    'hvd_device', None, 'Horovod`s equipment used for training')
flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
                    'file.')
flags.DEFINE_string('config_override', None, 'A pipeline_pb2.TrainEvalPipelineConfig text proto to '
      'override the config from `pipeline_config_path.')
flags.DEFINE_string("raport_file", default="summary.json",
                         help="Path to dlloger json")
flags.DEFINE_integer('num_steps', None, 'Number of train steps.')
flags.DEFINE_integer('batch_size', 32, 'Number of batch size.')
flags.DEFINE_boolean('eval_training_data', False,
                     'If training data should be evaluated for this job. Note '
                     'that one call only use this in eval-only mode, and '
                     '`checkpoint_dir` must be supplied.')
flags.DEFINE_integer('sample_1_of_n_eval_examples', 1, 'Will sample one of '
                     'every n eval input examples, where n is provided.')
flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample '
                     'one of every n train input examples for evaluation, '
                     'where n is provided. This is only used if '
                     '`eval_training_data` is True.')
flags.DEFINE_integer('eval_count', 1, 'How many times the evaluation should be run')
flags.DEFINE_string(
    'hparams_overrides', None, 'Hyperparameter overrides, '
    'represented as a string containing comma-separated '
    'hparam_name=value pairs.')
flags.DEFINE_string(
    'checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
    '`checkpoint_dir` is provided, this binary operates in eval-only mode, '
    'writing resulting metrics to `model_dir`.')
flags.DEFINE_string(
    'fine_tune_checkpoint', None, 'finetune checkpoint')
flags.DEFINE_string(
    'fine_tune_checkpoint_type', None, 'type of finetune, detection or classification')
flags.DEFINE_boolean(
    'allow_xla', False, 'Enable XLA compilation')
flags.DEFINE_boolean(
    'amp', False, 'Whether to enable AMP ops. When false, uses TF32 on A100 and FP32 on V100 GPUS.')
flags.DEFINE_boolean(
    'run_once', False, 'If running in eval-only mode, whether to run just '
    'one round of eval vs running continuously (default).'
)
flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool(
    'use_amp', False, 'If use amp, please set True.')
FLAGS = flags.FLAGS

class DLLoggerHook(tf.estimator.SessionRunHook):
  def __init__(self, global_batch_size, rank=-1):
    self.global_batch_size = global_batch_size
    self.rank = rank
    setup_dllogger(enabled=True, filename=FLAGS.raport_file, rank=rank)

  def after_create_session(self, session, coord):
    self.meters = {}
    warmup = 100
    self.meters['train_throughput'] = AverageMeter(warmup=warmup)

  def before_run(self, run_context):
    self.t0 = time.time()
    return tf.estimator.SessionRunArgs(fetches=['global_step:0', 'learning_rate:0'])

  def after_run(self, run_context, run_values):
    throughput = self.global_batch_size/(time.time() - self.t0)
    global_step, lr = run_values.results
    self.meters['train_throughput'].update(throughput)

  def end(self, session):
    summary = {
      'train_throughput': self.meters['train_throughput'].avg,
    }
    dllogger.log(step=tuple(), data=summary)

def _ensure_dir(log_dir):
  """Makes log dir if not existed."""
  if not tf.io.gfile.exists(log_dir):
    tf.io.gfile.makedirs(log_dir)

def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.use_performance and FLAGS.use_profiler:
    raise ValueError("You can only set use_profiler or use_performance, not at the same time, otherwise the e2e time will be worse")

  if FLAGS.use_horovod:
    import horovod.tensorflow as hvd
    global hvd
    hvd.init()
  if FLAGS.use_performance:
    from record_time import TimeHook, write_json
    global TimeHook
    global write_json

  train_hooks = []
  eval_hooks = []

  # Set profiler to train_hooks/eval_hooks
  if FLAGS.use_profiler and ((FLAGS.use_horovod and hvd.rank() == 0) or not FLAGS.use_horovod):
    timeline_hook = tf.estimator.ProfilerHook(save_steps=5, output_dir='profiler/')
    train_hooks.append(timeline_hook)
  # set performance e2e to hooks
  if FLAGS.use_performance and ((FLAGS.use_horovod and hvd.rank() == 0) or not FLAGS.use_horovod):
    time_hooks = TimeHook()
    train_hooks.append(time_hooks)

  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('pipeline_config_path')

  get_configs = config_util.get_configs_from_pipeline_file(FLAGS.pipeline_config_path,
                                           config_override=FLAGS.config_override)
  get_configs = config_util.merge_external_params_with_configs(
      get_configs, model_hparams.create_hparams(FLAGS.hparams_overrides))
  #update the number of training steps
  if FLAGS.num_steps:
      get_configs['train_config'].num_steps = FLAGS.num_steps
  if FLAGS.fine_tune_checkpoint:
      get_configs['train_config'].fine_tune_checkpoint = FLAGS.fine_tune_checkpoint
  if FLAGS.fine_tune_checkpoint_type:
      get_configs['train_config'].fine_tune_checkpoint_type = FLAGS.fine_tune_checkpoint_type
  if FLAGS.batch_size:
      get_configs['train_config'].batch_size = FLAGS.batch_size

  train_get_configs = get_configs['train_config']
  train_steps_ = train_get_configs.num_steps
  global_batch_size = train_get_configs.batch_size
  if FLAGS.use_horovod:
    global_batch_size *= hvd.size()

  session_config = tf.ConfigProto()


  if FLAGS.use_horovod:
    if FLAGS.hvd_device == 'mlu':
      session_config.mlu_options.visible_device_list = str(hvd.local_rank())
    elif FLAGS.hvd_device == 'gpu':
      session_config.gpu_options.visible_device_list = str(hvd.local_rank())
  if FLAGS.allow_xla:
    session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
  if ((FLAGS.use_horovod and hvd.rank() == 0) or (not FLAGS.use_horovod)):
      model_dir = FLAGS.model_dir
      _ensure_dir(model_dir)
  else:
    model_dir = None
  config = tf.estimator.RunConfig(model_dir=model_dir, session_config=session_config)

  train_and_eval_dict = model_lib.create_estimator_and_inputs(
      run_config=config,
      eval_count=FLAGS.eval_count,
      hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
      configs=get_configs,
      config_override=FLAGS.config_override,
      train_steps=FLAGS.num_steps,
      sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
      sample_1_of_n_eval_on_train_examples=(
          FLAGS.sample_1_of_n_eval_on_train_examples),
      use_horovod=FLAGS.use_horovod,
      use_amp=FLAGS.use_amp,
      fine_tune_checkpoint=FLAGS.fine_tune_checkpoint,
      fine_tune_checkpoint_type=FLAGS.fine_tune_checkpoint_type)
  estimator = train_and_eval_dict['estimator']
  train_input_fn = train_and_eval_dict['train_input_fn']
  eval_input_fns = train_and_eval_dict['eval_input_fns']
  eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
  predict_input_fn = train_and_eval_dict['predict_input_fn']
  train_steps = train_and_eval_dict['train_steps']


  if FLAGS.checkpoint_dir:
    if FLAGS.eval_training_data:
      name = 'training_data'
      input_fn = eval_on_train_input_fn
    else:
      name = 'validation_data'
      # The first eval input will be evaluated.
      input_fn = eval_input_fns[0]
    if FLAGS.run_once:
      estimator.evaluate(input_fn,
                         steps=None,
                         checkpoint_path=tf.train.latest_checkpoint(
                             FLAGS.checkpoint_dir))
    else:
      model_lib.continuous_eval(estimator, FLAGS.checkpoint_dir, input_fn,
                                train_steps, name)
  else:
    train_spec, eval_specs = model_lib.create_train_and_eval_specs(
        train_input_fn,
        eval_input_fns,
        eval_on_train_input_fn,
        predict_input_fn,
        train_steps,
        eval_on_train_data=False,
        use_horovod=FLAGS.use_horovod)

    if FLAGS.use_horovod:
        train_hooks.append(hvd.BroadcastGlobalVariablesHook(0))
        train_hooks.append(DLLoggerHook(hvd.size()*train_and_eval_dict['train_batch_size'], hvd.rank()))

    for x in range(FLAGS.eval_count):
      #whether do_train
      if FLAGS.do_train:
        estimator.train(train_input_fn,
                        hooks=train_hooks,
                        steps=train_steps // FLAGS.eval_count)

        if FLAGS.use_performance and ((FLAGS.use_horovod and hvd.rank() == 0) or not FLAGS.use_horovod):
            write_json("summary", global_batch_size, time_hooks.times)
      else:
        if ((FLAGS.use_horovod and hvd.rank() == 0) or (not FLAGS.use_horovod)):
            eval_input_fn = eval_input_fns[0]
            results = estimator.evaluate(eval_input_fn,
                               steps=None,
                               hooks=eval_hooks)

if __name__ == '__main__':
  tf.app.run()
