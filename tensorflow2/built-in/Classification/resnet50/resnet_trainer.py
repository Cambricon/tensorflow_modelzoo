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

"""Runs a ResNet model on the ImageNet dataset using custom training loops."""

import math
import os

# Import libraries
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from models import orbit
from models.common import distribute_utils
from models.modeling import performance
from models.utils.flags import core as flags_core
from models.utils.misc import keras_utils
from models.utils.misc import model_helpers
from models.resnet import common
from models.resnet import imagenet_preprocessing
from models.resnet import resnet_runnable
import time

flags.DEFINE_boolean(
    name="use_tf_function",
    default=True,
    help="Wrap the train and test step inside a " "tf.function.",
)
flags.DEFINE_boolean(
    name="single_l2_loss_op",
    default=False,
    help="Calculate L2_loss on concatenated weights, "
    "instead of using Keras per-layer L2 loss.",
)

flags.DEFINE_integer(name="num_mlus", default=1, help="The number of mlu devices.")

flags.DEFINE_boolean(
    name="use_horovod", default=False, help="Use hvd to train networks."
)

flags.DEFINE_string(
    "mode",
    "train_and_eval",
    help="Use mode to run `train_and_eval`, `train` and `eval`.",
)

flags.DEFINE_string(
    name="tf_mlu_thread_mode",
    default=None,
    help="Whether and how the MLU device uses its own threadpool.",
)

flags.DEFINE_integer(
    name="per_mlu_thread_count",
    default=0,
    help="The number of threads to use for MLU. Only valid when "
    "tf_mlu_thread_mode is not global.",
)

flags.DEFINE_float("base_learning_rate", "0.1", help="base learning rate.")

flags.DEFINE_boolean(name="use_performance", default=False, help="Use performance.")

flags.DEFINE_boolean(name="use_profiler", default=False, help="Use profiler.")

flags.DEFINE_boolean(name="use_amp", default=False, help="Use amp.")

flags.DEFINE_integer(
    "host_tracer_level",
    0,
    help="When using profiler for performance analysis, "
    "you need to set host tracer level.",
)

flags.DEFINE_integer(
    "device_tracer_level",
    0,
    help="When using profiler for performance analysis, "
    "you need to set device tracer level.",
)

flags.DEFINE_string(
    "profiler_dir",
    "./profiler",
    help="When using profiler for performance analysis,"
    "you need to set `profiler_dir` to save data",
)


def build_stats(runnable, time_callback):
    """Normalizes and returns dictionary of stats.

    Args:
      runnable: The module containing all the training and evaluation metrics.
      time_callback: Time tracking callback instance.

    Returns:
      Dictionary of normalized results.
    """
    stats = {}

    if not runnable.flags_obj.skip_eval:
        stats["eval_loss"] = runnable.test_loss.result().numpy()
        stats["eval_acc"] = runnable.test_accuracy.result().numpy()

        stats["train_loss"] = runnable.train_loss.result().numpy()
        stats["train_acc"] = runnable.train_accuracy.result().numpy()

    if time_callback:
        timestamp_log = time_callback.timestamp_log
        stats["step_timestamp_log"] = timestamp_log
        stats["train_finish_time"] = time_callback.train_finish_time
        if time_callback.epoch_runtime_log:
            stats["avg_exp_per_second"] = time_callback.average_examples_per_second

    return stats


def get_num_train_iterations(flags_obj):
    """Returns the number of training steps, train and test epochs."""
    train_steps = imagenet_preprocessing.NUM_IMAGES["train"] // flags_obj.batch_size
    if flags_obj.use_horovod:
        train_steps = train_steps // hvd.size()
    train_epochs = flags_obj.train_epochs

    if flags_obj.train_steps:
        train_steps = flags_obj.train_steps
        train_epochs = 1

    eval_steps = math.ceil(
        1.0 * imagenet_preprocessing.NUM_IMAGES["validation"] / flags_obj.batch_size
    )

    return train_steps, train_epochs, eval_steps


def run(flags_obj):
    """Run ResNet ImageNet training and eval loop using custom training loops.

    Args:
      flags_obj: An object containing parsed flag values.

    Raises:
      ValueError: If fp16 is passed as it is not currently supported.

    Returns:
      Dictionary of training and eval stats.
    """
    if flags_obj.use_horovod:
        import horovod.tensorflow as hvd

        global hvd
        hvd.init()
    if flags_obj.use_amp:
        flags_obj.dtype = "fp16"

    keras_utils.set_session_config()
    performance.set_mixed_precision_policy(flags_core.get_tf_dtype(flags_obj))

    if flags_obj.use_horovod:
        if flags_obj.num_mlus > 0 and flags_obj.num_gpus == 0:
            mlus = tf.config.experimental.list_physical_devices("MLU")
            if mlus:
                tf.config.experimental.set_visible_devices(
                    mlus[hvd.local_rank()], "MLU"
                )
        elif flags_obj.num_mlus == 0 and flags_obj.num_gpus > 0:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if gpus:
                tf.config.experimental.set_visible_devices(
                    gpus[hvd.local_rank()], "GPU"
                )
        else:
            raise ValueError(
                "Horovod can only be used when only one of gpu or mlu is greater than 0"
            )

    if tf.config.list_physical_devices("MLU"):
        if flags_obj.tf_mlu_thread_mode:
            common.set_thread_mode_and_count(
                mlu_thread_mode=flags_obj.tf_mlu_thread_mode,
                per_mlu_thread_count=flags_obj.per_mlu_thread_count,
                interop_threads=flags_obj.inter_op_parallelism_threads,
                intraop_threads=flags_obj.intra_op_parallelism_threads,
            )
    elif tf.config.list_physical_devices("GPU"):
        if flags_obj.tf_gpu_thread_mode:
            keras_utils.set_gpu_thread_mode_and_count(
                per_gpu_thread_count=flags_obj.per_gpu_thread_count,
                gpu_thread_mode=flags_obj.tf_gpu_thread_mode,
                num_gpus=flags_obj.num_gpus,
                datasets_num_private_threads=flags_obj.datasets_num_private_threads,
            )
        common.set_cudnn_batchnorm_mode()

    data_format = flags_obj.data_format
    if data_format is None:
        data_format = "channels_last"
    tf.keras.backend.set_image_data_format(data_format)

    strategy = distribute_utils.get_distribution_strategy(
        distribution_strategy=flags_obj.distribution_strategy,
        num_gpus=flags_obj.num_gpus,
        num_mlus=flags_obj.num_mlus,
        all_reduce_alg=flags_obj.all_reduce_alg,
        num_packs=flags_obj.num_packs,
        tpu_address=flags_obj.tpu,
    )

    per_epoch_steps, train_epochs, eval_steps = get_num_train_iterations(flags_obj)
    if flags_obj.steps_per_loop is None:
        steps_per_loop = per_epoch_steps
    elif flags_obj.steps_per_loop > per_epoch_steps:
        steps_per_loop = per_epoch_steps
        logging.warn(
            "Setting steps_per_loop to %d to respect epoch boundary.", steps_per_loop
        )
    else:
        steps_per_loop = flags_obj.steps_per_loop

    logging.info(
        "Training %d epochs, each epoch has %d steps, "
        "total steps: %d; Eval %d steps",
        train_epochs,
        per_epoch_steps,
        train_epochs * per_epoch_steps,
        eval_steps,
    )

    if flags_obj.use_performance:
        from record_time import write_json

        global write_json
    time_callback = None
    if (flags_obj.use_horovod and hvd.rank() == 0) or not flags_obj.use_horovod:
        time_callback = keras_utils.TimeHistory(
            flags_obj.batch_size,
            flags_obj.log_steps,
            logdir=flags_obj.model_dir if flags_obj.enable_tensorboard else None,
        )
    with distribute_utils.get_strategy_scope(strategy):
        runnable = resnet_runnable.ResnetRunnable(
            flags_obj,
            time_callback,
            per_epoch_steps,
            use_performance=flags_obj.use_performance,
            use_profiler=flags_obj.use_profiler,
            hvd=hvd if flags_obj.use_horovod else None,
            profiler_dir=flags_obj.profiler_dir,
            host_tracer_level=flags_obj.host_tracer_level,
            device_tracer_level=flags_obj.device_tracer_level,
            base_learning_rate=flags_obj.base_learning_rate,
        )

    eval_interval = flags_obj.epochs_between_evals * per_epoch_steps
    checkpoint_interval = (
        steps_per_loop * 5 if flags_obj.enable_checkpoint_and_export else None
    )
    summary_interval = steps_per_loop if flags_obj.enable_tensorboard else None

    checkpoint_manager = None
    checkpoint_manager = tf.train.CheckpointManager(
        runnable.checkpoint,
        directory=flags_obj.model_dir,
        max_to_keep=10,
        step_counter=runnable.global_step,
        checkpoint_interval=checkpoint_interval,
    )

    resnet_controller = orbit.Controller(
        strategy=strategy,
        trainer=runnable,
        evaluator=runnable if not flags_obj.skip_eval else None,
        global_step=runnable.global_step,
        steps_per_loop=steps_per_loop,
        checkpoint_manager=checkpoint_manager,
        summary_interval=summary_interval,
        summary_dir=flags_obj.model_dir
        if (flags_obj.use_horovod and hvd.rank() == 0) or not flags_obj.use_horovod
        else None,
        eval_summary_dir=os.path.join(flags_obj.model_dir, "eval"),
        hvd=hvd if flags_obj.use_horovod else None,
    )

    if (flags_obj.use_horovod and hvd.rank() == 0) or not flags_obj.use_horovod:
        time_callback.on_train_begin()
    if flags_obj.mode == "train_and_eval":
        resnet_controller.train_and_evaluate(
            train_steps=per_epoch_steps * train_epochs,
            eval_steps=eval_steps,
            eval_interval=eval_interval,
        )
    elif flags_obj.mode == "train":
        resnet_controller.train(steps=per_epoch_steps * train_epochs)
    elif flags_obj.mode == "eval":
        resnet_controller.evaluate(steps=eval_steps)
    else:
        raise ValueError("Must be set mode, `train_and_eval` or `train` or `eval`.")
    if (flags_obj.use_horovod and hvd.rank() == 0) or not flags_obj.use_horovod:
        time_callback.on_train_end()
    if flags_obj.use_performance:
        global_batch_size = (
            flags_obj.batch_size
            if not flags_obj.use_horovod
            else flags_obj.batch_size * hvd.size()
        )
        if (flags_obj.use_horovod and hvd.rank() == 0) or not flags_obj.use_horovod:
            write_json("summary", global_batch_size, runnable.time_list)

    stats = build_stats(runnable, time_callback)
    return stats


def main(_):
    model_helpers.apply_clean(flags.FLAGS)
    stats = run(flags.FLAGS)
    logging.info("Run stats:\n%s", stats)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    common.define_keras_flags()
    app.run(main)
