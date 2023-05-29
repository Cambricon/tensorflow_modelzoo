# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#
# author: Tomasz Grel (tgrel@nvidia.com)


from absl import app, flags
import os
import sys

# Define the flags first before importing TensorFlow.
# Otherwise, enabling XLA-Lite would be impossible with a command-line flag
def define_command_line_flags():
    flags.DEFINE_enum("mode", default="train", enum_values=['inference', 'eval', 'train'],
                      help='Choose "train" to train the model, "inference" to benchmark inference'
                      ' and "eval" to run validation')
    flags.DEFINE_float("learning_rate", default=24, help="Learning rate")
    flags.DEFINE_integer("batch_size", default=64 * 1024, help="Batch size used for training")
    flags.DEFINE_integer("valid_batch_size", default=64 * 1024, help="Batch size used for validation")
    flags.DEFINE_bool("run_eagerly", default=False, help="Disable all tf.function decorators for debugging")

    flags.DEFINE_bool("dummy_model", default=False, help="Use a dummy model for benchmarking and debugging")
    flags.DEFINE_bool("dummy_embedding", default=False, help="")

    flags.DEFINE_list("top_mlp_dims", [1024, 1024, 512, 256, 1], "Linear layer sizes for the top MLP")
    flags.DEFINE_list("bottom_mlp_dims", [512, 256, 128], "Linear layer sizes for the bottom MLP")

    flags.DEFINE_enum("optimizer", default="sgd", enum_values=['sgd', 'adam'],
                      help='The optimization algorithm to be used.')

    flags.DEFINE_string("save_checkpoint_path", default=None,
                        help="Path to which to save a checkpoint file at the end of the training")
    flags.DEFINE_string("restore_checkpoint_path", default=None,
                        help="Path from which to restore a checkpoint before training")

    flags.DEFINE_enum("dataset_type", default="raw", enum_values=['raw', 'synthetic'],
                      help='The type of the dataset to use')
    flags.DEFINE_integer("num_numerical_features", default=13,
                      help='Number of numerical features to be read from the dataset. '
                           'If set to 0, then no numerical features will be loaded '
                           'and the Bottom MLP will not be evaluated')

    flags.DEFINE_integer('synthetic_dataset_train_batches', default=64008,
                         help='Number of training batches in the synthetic dataset')
    flags.DEFINE_integer('synthetic_dataset_valid_batches', default=1350,
                         help='Number of validation batches in the synthetic dataset')
    flags.DEFINE_list('synthetic_dataset_cardinalities', default=26*[1000],
                         help='Number of categories for each embedding table of the synthetic dataset')

    flags.DEFINE_bool("amp", default=False, help="Enable automatic mixed precision")
    flags.DEFINE_bool("use_amp", default=False, help="Enable automatic mixed precision on MLU")
    flags.DEFINE_bool("enable_xla", default=False, help="Whether to enable XLA auto jit compilation")

    flags.DEFINE_integer("loss_scale", default=1024, help="Static loss scale to use with mixed precision training")

    flags.DEFINE_integer("prefetch_batches", default=10,
                         help="The number of batches to prefetch for the dataloader")

    flags.DEFINE_integer("auc_thresholds", default=8000,
                         help="Number of thresholds for the AUC computation")

    flags.DEFINE_integer("epochs", default=1, help="Number of epochs to train for")
    flags.DEFINE_integer("max_steps", default=-1, help="Stop the training/inference after this many optimiation steps")

    flags.DEFINE_string("embedding_type", default="split_embedding",
                        help="Embedding type to use, possible choices: embedding, split_embedding")
    flags.DEFINE_bool("embedding_trainable", default=True, help="If True the embeddings will be trainable, otherwise frozen")

    flags.DEFINE_string("dot_interaction", default="custom_cuda",
                        help="Dot interaction implementation to use, possible choices: custom_cuda, tensorflow, dummy")

    flags.DEFINE_string("dataset_path", default=None,
                        help="Path to the JSON file with the sizes of embedding tables")

    flags.DEFINE_integer("embedding_dim", default=128, help='Number of columns in the embedding tables')

    flags.DEFINE_integer("evals_per_epoch", default=1, help='Number of evaluations per epoch')
    flags.DEFINE_float("print_freq", default=1000, help='Number of steps between debug prints')

    flags.DEFINE_integer("warmup_steps", default=8000,
                        help='Number of steps over which to linearly increase the LR at the beginning')
    flags.DEFINE_integer("decay_start_step", default=48000, help='Optimization step at which to start the poly LR decay')
    flags.DEFINE_integer("decay_steps", default=24000, help='Number of steps over which to decay from base LR to 0')

    flags.DEFINE_integer("profiler_start_step", default=None, help='Step at which to start profiling')
    flags.DEFINE_integer("profiler_steps", default=1, help='Number of profiling steps')
    flags.DEFINE_integer("profiled_rank", default=1, help='Rank to profile')

    flags.DEFINE_integer("inter_op_parallelism", default=None, help='Number of inter op threads')
    flags.DEFINE_integer("intra_op_parallelism", default=None, help='Number of intra op threads')

    flags.DEFINE_integer("tf_mlu_memory_limit_gb", default=24,
                         help='Gigabytes of MLU memory reserved for TensorFlow. Only applied in multiMLU/multiNode to leave'
                              ' enough memory for NCCL to operate properly.')

    flags.DEFINE_bool("data_parallel_bottom_mlp", default=False, help="Run the bottom MLP in data-parallel mode")
    flags.DEFINE_bool("experimental_columnwise_split", default=False,
                      help="Enable slicing individual embedding tables across multiple devices")

    flags.DEFINE_string("log_path", default='dlrm_tf_log.json', help="Path to JSON file for storing benchmark results")

    flags.DEFINE_bool("use_horovod", default=False, help="If use horovod, please set True.")
    flags.DEFINE_bool("use_profiler", default=False, help="Use profiler or not. It should be True or False.")
    flags.DEFINE_bool("use_performance", default=False, help="use performance tools to get fps or not.")

    flags.DEFINE_bool("use_mlus", default=True,
                         help="Use of MLUs")
    flags.DEFINE_bool("use_gpus", default=False,
                         help="Use of GPUs")
    flags.DEFINE_bool("skip_eval", default=False, help="Whether to skip_eval per epoch")
    flags.DEFINE_integer("benchmark_warmup_steps", default=100,
                        help='Number of steps over which to linearly increase the LR at the benchmark')
define_command_line_flags()

FLAGS = flags.FLAGS

app.define_help_flags()
app.parse_flags_with_usage(sys.argv)
FLAGS.amp=FLAGS.use_amp
if FLAGS.enable_xla:
    # The fusion strategy using fusible for dlrm is the best choice at present
    os.environ['TF_XLA_FLAGS'] = (os.environ.get("TF_XLA_FLAGS", "") +
        " --tf_xla_auto_jit=fusible --tf_xla_enable_lazy_compilation=false --tf_xla_async_io_level=0 ")

import time
from models.lr_scheduler import LearningRateScheduler
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from models.utils import IterTimer, init_logging, dist_print
from models.dataloader import create_input_pipelines
from models.model import Dlrm, DummyDlrm, DlrmTrainer, evaluate, DataParallelSplitter
import horovod.tensorflow as hvd
from tensorflow.keras.mixed_precision import LossScaleOptimizer
import dllogger

def init_tf(FLAGS):
    """
    Set global options for TensorFlow
    """

    if FLAGS.use_mlus:
        dev = 'MLU'
    elif FLAGS.use_gpus:
        dev = 'GPU'
    else:
        dev = 'CPU'
    devs = tf.config.experimental.list_physical_devices(dev)

    if devs:
        tf.config.experimental.set_visible_devices(devs[hvd.local_rank()], dev)

    if FLAGS.amp:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)

    tf.config.run_functions_eagerly(FLAGS.run_eagerly)

    if FLAGS.inter_op_parallelism:
        tf.config.threading.set_inter_op_parallelism_threads(FLAGS.inter_op_parallelism)

    if FLAGS.intra_op_parallelism:
        tf.config.threading.set_intra_op_parallelism_threads(FLAGS.intra_op_parallelism)

def compute_eval_points(train_batches, evals_per_epoch):
    eval_points = np.linspace(0, train_batches - 1, evals_per_epoch + 1)[1:]
    eval_points = np.round(eval_points).tolist()
    return eval_points


def inference_benchmark(validation_pipeline, dlrm, timer, splitter, FLAGS):
    if FLAGS.max_steps == -1:
        FLAGS.max_steps = 1000

    _, _, latencies = evaluate(validation_pipeline, dlrm,
                               timer, auc_thresholds=None,
                               data_parallel_splitter=splitter,
                               max_steps=FLAGS.max_steps)

    # don't benchmark the first few warmup steps
    latencies = latencies[10:]
    result_data = {
        'mean_inference_throughput': FLAGS.valid_batch_size / np.mean(latencies),
        'mean_inference_latency': np.mean(latencies)
    }

    for percentile in [90, 95, 99]:
        result_data[f'p{percentile}_inference_latency'] = np.percentile(latencies, percentile)
    dllogger.log(data=result_data, step=tuple())


def main(argv):
    if FLAGS.experimental_columnwise_split and not FLAGS.data_parallel_bottom_mlp and FLAGS.num_numerical_features > 0:
        raise ValueError('Currently you when using the --experimenal_columnwise_split option '
                         'you must either set --data_parallel_bottom_mlp or --num_numerical_features=0')

    if FLAGS.batch_size != FLAGS.valid_batch_size:
        FLAGS.valid_batch_size = FLAGS.batch_size

    if FLAGS.use_performance:
        from record_time import TimeHook, write_json
        global TimeHook
        global write_json
    hvd.init()
    init_logging(log_path=FLAGS.log_path, FLAGS=FLAGS)
    init_tf(FLAGS)

    train_pipeline, validation_pipeline, dataset_metadata, multi_mlu_metadata = create_input_pipelines(FLAGS)

    if FLAGS.dummy_model:
        dlrm = DummyDlrm(FLAGS=FLAGS, dataset_metadata=dataset_metadata,
                         multi_mlu_metadata=multi_mlu_metadata)
    else:
        dlrm = Dlrm(FLAGS=FLAGS, dataset_metadata=dataset_metadata,
                    multi_mlu_metadata=multi_mlu_metadata)

    if FLAGS.optimizer == 'sgd':
        embedding_optimizer = tf.keras.optimizers.legacy.SGD(lr=FLAGS.learning_rate, momentum=0)
        if FLAGS.amp:
            embedding_optimizer = LossScaleOptimizer(embedding_optimizer,
                                                     initial_scale=FLAGS.loss_scale,
                                                     dynamic=False)
        mlp_optimizer = embedding_optimizer
        optimizers = [mlp_optimizer]

    elif FLAGS.optimizer == 'adam':
        embedding_optimizer = tfa.optimizers.LazyAdam(lr=FLAGS.learning_rate)
        mlp_optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
        if FLAGS.amp:
            embedding_optimizer = LossScaleOptimizer(embedding_optimizer,
                                                     initial_scale=FLAGS.loss_scale,
                                                     dynamic=False)
            mlp_optimizer = LossScaleOptimizer(mlp_optimizer,
                                               initial_scale=FLAGS.loss_scale,
                                               dynamic=False)
        optimizers = [mlp_optimizer, embedding_optimizer]

    scheduler = LearningRateScheduler(optimizers,
                                      warmup_steps=FLAGS.warmup_steps,
                                      base_lr=FLAGS.learning_rate,
                                      decay_start_step=FLAGS.decay_start_step,
                                      decay_steps=FLAGS.decay_steps)

    timer = IterTimer(train_batch_size=FLAGS.batch_size, test_batch_size=FLAGS.valid_batch_size,
                      optimizer=embedding_optimizer, print_freq=FLAGS.print_freq, enabled=hvd.rank() == 0,
                      benchmark_warmup_steps=FLAGS.benchmark_warmup_steps)

    splitter = DataParallelSplitter(batch_size=FLAGS.batch_size)

    dlrm.maybe_restore_checkpoint(FLAGS.restore_checkpoint_path)

    if FLAGS.mode == 'inference':
        inference_benchmark(validation_pipeline, dlrm, timer, splitter, FLAGS)
        return

    elif FLAGS.mode == 'eval':
        test_auc, test_loss, _ = evaluate(validation_pipeline, dlrm,
                                          timer, auc_thresholds=FLAGS.auc_thresholds,
                                          data_parallel_splitter=splitter)
        dist_print(f'Evaluation completed, AUC: {test_auc:.6f}, test_loss: {test_loss:.6f}')
        return

    eval_points = compute_eval_points(train_batches=len(train_pipeline),
                                      evals_per_epoch=FLAGS.evals_per_epoch)

    trainer = DlrmTrainer(dlrm, embedding_optimizer=embedding_optimizer,
                          mlp_optimizer=mlp_optimizer, amp=FLAGS.amp,
                          lr_scheduler=scheduler)

    best_auc = 0
    train_begin = time.time()
    if FLAGS.use_performance and hvd.rank() == 0:
        times = []
    for epoch in range(FLAGS.epochs):
        for step, ((numerical_features, categorical_features), labels) in enumerate(train_pipeline):
            if FLAGS.use_profiler:
                if step == FLAGS.profiler_start_step and hvd.rank() == FLAGS.profiled_rank:
                    tf.profiler.experimental.start(FLAGS.save_checkpoint_path)

                if step == FLAGS.profiler_start_step + FLAGS.profiler_steps and hvd.rank() == FLAGS.profiled_rank:
                    tf.profiler.experimental.stop()

            labels = splitter(labels)
            if FLAGS.data_parallel_bottom_mlp:
                numerical_features = splitter(numerical_features)
            if FLAGS.use_performance and hvd.rank() == 0:
                start_time = time.time()
            loss = trainer.train_step(numerical_features, categorical_features, labels)
            if FLAGS.use_performance and hvd.rank() == 0:
                times.append(time.time() - start_time)

            timer.step_train(loss=loss)

            if FLAGS.max_steps != -1 and step > FLAGS.max_steps:
                dist_print(f'Max steps of {FLAGS.max_steps} reached, exiting')
                break

            if not FLAGS.skip_eval:
                if step in eval_points:
                    test_auc, test_loss, _ = evaluate(validation_pipeline, dlrm,
                                                      timer, FLAGS.auc_thresholds,
                                                      data_parallel_splitter=splitter)
                    dist_print(f'Evaluation completed, AUC: {test_auc:.6f}, test_loss: {test_loss:.6f}')
                    timer.test_idx = 0
                    best_auc = max(best_auc, test_auc)
    if FLAGS.use_performance and hvd.rank() == 0:
        write_json("summary", FLAGS.batch_size, times)

    elapsed = time.time() - train_begin
    if not FLAGS.use_performance:
        dlrm.maybe_save_checkpoint(FLAGS.save_checkpoint_path)

    if hvd.rank() == 0:
        dist_print(f'Training run completed, elapsed: {elapsed:.0f} [s]')
        results = {
            'throughput': FLAGS.batch_size / timer.mean_train_time(),
            'mean_step_time_ms': timer.mean_train_time() * 1000,
            'auc': best_auc
        }
        dllogger.log(data=results, step=tuple())


if __name__ == '__main__':
    app.run(main)
