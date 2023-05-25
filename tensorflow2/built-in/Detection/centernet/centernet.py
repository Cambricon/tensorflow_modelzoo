# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

r"""Creates and runs TF2 object detection models.

For local training/evaluation run:
PIPELINE_CONFIG_PATH=path/to/pipeline.config
MODEL_DIR=/tmp/model_outputs
NUM_TRAIN_STEPS=10000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python model_main_tf2.py -- \
  --model_dir=$MODEL_DIR --num_train_steps=$NUM_TRAIN_STEPS \
  --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
  --pipeline_config_path=$PIPELINE_CONFIG_PATH \
  --alsologtostderr
"""
from absl import flags

import os
import sys
cur_path = os.getcwd()
models_path = cur_path + "/models/"
sys.path.append(models_path)

import tensorflow.compat.v2 as tf
import model_lib_v2
from official.modeling import performance

flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
                    'file.')
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
flags.DEFINE_bool('eval_on_train_data', False, 'Enable evaluating on train '
                  'data (only supported in distributed training).')
flags.DEFINE_integer('sample_1_of_n_eval_examples', None, 'Will sample one of '
                     'every n eval input examples, where n is provided.')
flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample '
                     'one of every n train input examples for evaluation, '
                     'where n is provided. This is only used if '
                     '`eval_training_data` is True.')
flags.DEFINE_string(
    'model_dir', None, 'Path to output model directory '
                       'where event and checkpoint files will be written.')
flags.DEFINE_string(
    'checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
    '`checkpoint_dir` is provided, this binary operates in read-only mode, '
    'writing resulting metrics to `model_dir`.')

flags.DEFINE_integer('eval_timeout', 3600, 'Number of seconds to wait for an'
                     'evaluation checkpoint before exiting.')

flags.DEFINE_bool('use_tpu', False, 'Whether the job is executing on a TPU.')
flags.DEFINE_string(
    'tpu_name',
    default=None,
    help='Name of the Cloud TPU for Cluster Resolvers.')
flags.DEFINE_integer(
    'num_workers', 1, 'When num_workers > 1, training uses '
    'MultiWorkerMirroredStrategy. When num_workers = 1 it uses '
    'MirroredStrategy.')
flags.DEFINE_integer(
    'checkpoint_every_n', 1000, 'Integer defining how often we checkpoint.')
flags.DEFINE_boolean('record_summaries', True,
                     ('Whether or not to record summaries during'
                      ' training.'))
flags.DEFINE_integer('batch_size', None, 'The number of batch_size.')
flags.DEFINE_integer('num_steps_per_iter', None, 'The number of steps per iteration.')
flags.DEFINE_bool('do_train', False, 'Whether to run training.')
flags.DEFINE_bool('use_gpus', False, 'Whether to run training on GPU devices.')
flags.DEFINE_bool('use_horovod', False, 'Whether to use horovod to train network.')
flags.DEFINE_string(
    'finetune_ckpt', None, 'Checkpoint to finetune some steps during train.')
flags.DEFINE_bool(
    'load_latest_ckpt_continuously', True, 'Whether to load the latest ckpt '
    'continuously, otherwise only load the latest ckpt once for inference.')
flags.DEFINE_bool(
    'use_amp',
    default=False,
    help='If use amp, please set True.')
flags.DEFINE_bool(
    'use_profiler', False, 'Use profiler or not. It should be True or False')
flags.DEFINE_bool(
    'use_performance', False, 'Use performance test tools or not. It should be True or False')
flags.DEFINE_integer(
    'host_tracer_level', 2, 'When using profiler for performance analysis, '
    'you need to set host tracer level.')
flags.DEFINE_integer(
    'device_tracer_level', 1, 'When using profiler for performance analysis, '
    'you need to set device tracer level.')

FLAGS = flags.FLAGS

def get_available_devs(device_type='MLU'):
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == device_type]

def main(unused_argv):
  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('pipeline_config_path')
  tf.config.set_soft_device_placement(True)

  if FLAGS.checkpoint_dir and (not FLAGS.do_train):
    model_lib_v2.eval_continuously(
        pipeline_config_path=FLAGS.pipeline_config_path,
        model_dir=FLAGS.model_dir,
        train_steps=FLAGS.num_train_steps,
        sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
        sample_1_of_n_eval_on_train_examples=(
            FLAGS.sample_1_of_n_eval_on_train_examples),
        checkpoint_dir=FLAGS.checkpoint_dir,
        load_latest_ckpt_continuously=FLAGS.load_latest_ckpt_continuously,
        wait_interval=300, timeout=FLAGS.eval_timeout, use_amp=FLAGS.use_amp)
  else:
    if FLAGS.use_horovod:
      import horovod.tensorflow as hvd
      global hvd
      hvd.init()

    if FLAGS.use_horovod:
      if FLAGS.use_gpus:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
          tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
      else:
        mlus = tf.config.experimental.list_physical_devices('MLU')
        if mlus:
          tf.config.experimental.set_visible_devices(mlus[hvd.local_rank()], 'MLU')

    if FLAGS.use_tpu:
      # TPU is automatically inferred if tpu_name is None and
      # we are running under cloud ai-platform.
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu_name)
      tf.config.experimental_connect_to_cluster(resolver)
      tf.tpu.experimental.initialize_tpu_system(resolver)
      strategy = tf.distribute.experimental.TPUStrategy(resolver)
    elif FLAGS.num_workers > 1:
      strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    else:
      if FLAGS.use_horovod:
        if FLAGS.use_gpus:
          strategy = tf.compat.v2.distribute.OneDeviceStrategy("device:GPU:0")
        else:
          strategy = tf.compat.v2.distribute.OneDeviceStrategy("device:MLU:0")
      else:
        device_type = 'GPU' if FLAGS.use_gpus else 'MLU'
        local_devices = get_available_devs(device_type)
        strategy = tf.compat.v2.distribute.MirroredStrategy(local_devices)

    with strategy.scope():
      model_lib_v2.train_loop(
          pipeline_config_path=FLAGS.pipeline_config_path,
          model_dir=FLAGS.model_dir,
          train_steps=FLAGS.num_train_steps,
          use_tpu=FLAGS.use_tpu,
          checkpoint_every_n=FLAGS.checkpoint_every_n,
          record_summaries=FLAGS.record_summaries,
          batch_size_override=FLAGS.batch_size,
          finetune_ckpt=FLAGS.finetune_ckpt,
          use_profiler=FLAGS.use_profiler,
          host_tracer_level=FLAGS.host_tracer_level,
          device_tracer_level=FLAGS.device_tracer_level,
          num_steps_per_iteration=FLAGS.num_steps_per_iter,
          use_performance=FLAGS.use_performance,
          use_amp=FLAGS.use_amp)

if __name__ == '__main__':
  tf.compat.v1.app.run()
