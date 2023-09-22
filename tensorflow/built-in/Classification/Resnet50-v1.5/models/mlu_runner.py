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

from __future__ import print_function

import os
import multiprocessing
import warnings

import tensorflow as tf
import numpy as np

import mlu_resnet

from utils import hooks
import mlu_data_utils

from runtime import runner_utils

import dllogger

__all__ = [
    'Runner',
]

class Runner(object):

    def __init__(
        self,
        # ========= Model HParams ========= #
        n_classes=1001,
        architecture='resnet50',
        input_format='NHWC',  # NCHW or NHWC
        compute_format='NCHW',  # NCHW or NHWC
        dtype=tf.float32,  # tf.float32 or tf.float16
        n_channels=3,
        height=224,
        width=224,
        distort_colors=False,
        model_dir=None,
        log_dir=None,
        data_dir=None,
        data_idx_dir=None,
        weight_init="fan_out",

        # ======= Optimization HParams ======== #
        use_xla=False,
        use_tf_amp=False,
        use_dali=False,
        device_memory_fraction=1.0,
        device="MLU",
        device_id=0,

        # ======== Debug Flags ======== #
        debug_verbosity=0,
        seed=None,
        use_horovod=False,
        hvd_finetune_test=False,

        # ======== Performance Flags ======== #
        use_profiler=False,
        use_performance=False,
    ):

        self.device = device
        self.use_horovod_ = use_horovod
        self.use_profiler_ = use_profiler
        self.use_performance_ = use_performance

        if self.use_profiler_ and self.use_performance_:
            raise ValueError("You can only set use_profiler or use_performance, not at the same time, otherwise the e2e time will be worse")

        if dtype not in [tf.float32, tf.float16]:
            raise ValueError("Unknown dtype received: %s (allowed: `tf.float32` and `tf.float16`)" % dtype)

        if compute_format not in ["NHWC", 'NCHW']:
            raise ValueError("Unknown `compute_format` received: %s (allowed: ['NHWC', 'NCHW'])" % compute_format)

        if input_format not in ["NHWC", 'NCHW']:
            raise ValueError("Unknown `input_format` received: %s (allowed: ['NHWC', 'NCHW'])" % input_format)

        if n_channels not in [1, 3]:
            raise ValueError("Unsupported number of channels: %d (allowed: 1 (grayscale) and 3 (color))" % n_channels)

        tf_seed = 2 * seed if seed is not None else None
        if self.use_horovod_:
          import horovod.tensorflow as hvd
          global hvd
          tf_seed = 2 * (seed + hvd.rank()) if seed is not None else None

        if self.use_performance_:
          from record_time import TimeHook, write_json
          global TimeHook
          global write_json

        # Note that hvd_finetune_test is only used for internal hvd finetune(from 1-card training ckpt) accuracy test。
        self.hvd_finetune_test = hvd_finetune_test

        model_hparams = tf.contrib.training.HParams(
            width=height,
            height=width,
            n_channels=n_channels,
            n_classes=n_classes,
            dtype=dtype,
            input_format=input_format,
            compute_format=compute_format,
            distort_colors=distort_colors,
            seed=tf_seed
        )

        num_preprocessing_threads = 10 if not use_dali else 4
        run_config_performance = tf.contrib.training.HParams(
            num_preprocessing_threads=num_preprocessing_threads,
            use_tf_amp=use_tf_amp,
            use_xla=use_xla,
            use_dali=use_dali,
            device_memory_fraction=device_memory_fraction,
            device_id=device_id
        )

        run_config_additional = tf.contrib.training.HParams(
            model_dir=model_dir,
            log_dir=log_dir if (not self.use_horovod_) or (self.use_horovod_ and hvd.rank() == 0) else None,
            data_dir=data_dir,
            data_idx_dir=data_idx_dir,
            num_preprocessing_threads=num_preprocessing_threads,
            hvd_finetune_test=self.hvd_finetune_test
        )

        self.run_hparams = Runner._build_hparams(model_hparams, run_config_additional, run_config_performance)

        self._model_name = architecture
        architecture = mlu_resnet.model_architectures[architecture]

        self._model = mlu_resnet.ResnetModel(
            model_name=self._model_name,
            n_classes=model_hparams.n_classes,
            layers_count=architecture["layers"],
            layers_depth=architecture["widths"],
            expansions=architecture["expansions"],
            input_format=model_hparams.input_format,
            compute_format=model_hparams.compute_format,
            dtype=model_hparams.dtype,
            weight_init=weight_init,
            use_dali=use_dali,
            cardinality=architecture['cardinality'] if 'cardinality' in architecture else 1,
            use_se=architecture['use_se'] if 'use_se' in architecture else False,
            se_ratio=architecture['se_ratio'] if 'se_ratio' in architecture else 1,
            use_hvd=self.use_horovod_,
            device=self.device,
        )

        if self.run_hparams.seed is not None:
            tf.set_random_seed(self.run_hparams.seed)

        self.training_logging_hook = None
        self.eval_logging_hook = None

    @staticmethod
    def _build_hparams(*args):

        hparams = tf.contrib.training.HParams()

        for _hparams in args:
            if not isinstance(_hparams, tf.contrib.training.HParams):
                raise ValueError("Non valid HParams argument object detected:", _hparams)

            for key, val in _hparams.values().items():
                try:
                    hparams.add_hparam(name=key, value=val)

                except ValueError:
                    warnings.warn(
                        "the parameter `{}` already exists - existing value: {} and duplicated value: {}".format(
                            key, hparams.get(key), val
                        )
                    )

        return hparams

    @staticmethod
    def _get_global_batch_size(worker_batch_size, use_hvd):

        if use_hvd:
            return worker_batch_size * hvd.size()
        else:
            return worker_batch_size

    def _get_session_config(self, mode, use_xla, use_dali, device_memory_fraction, device_id=0, use_hvd=False):

        if mode not in ["train", 'validation', 'benchmark', 'inference']:
            raise ValueError(
                "Unknown mode received: %s (allowed: 'train', 'validation', 'benchmark', 'inference')" % mode
            )

        config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)

        # Limit available GPU memory (tune the size)
        # MLU use imagenet data instead of dali
        if use_dali and False:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=device_memory_fraction)
            config = tf.ConfigProto(gpu_options=gpu_options)
            config.gpu_options.allow_growth = False
        else:
            config = tf.ConfigProto()
            if self.device.lower() == "gpu":
                config.gpu_options.allow_growth = True

        config.allow_soft_placement = True
        config.log_device_placement = False

        if self.device.lower() == "mlu":
            config.mlu_options.visible_device_list = str(device_id)
        if self.device.lower() == "gpu":
            config.gpu_options.visible_device_list = str(device_id)

        if use_hvd:
            if self.device.lower() == "mlu":
                config.mlu_options.visible_device_list = str(hvd.local_rank())
            if self.device.lower() == "gpu":
                config.gpu_options.visible_device_list = str(hvd.local_rank())

        if use_xla:
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        if self.device.lower() == "gpu":
            config.gpu_options.force_gpu_compatible = True  # Force pinned memory

        # Bug - disable bn+relu fusion
        from tensorflow.core.protobuf import rewriter_config_pb2
        config.graph_options.rewrite_options.remapping = (rewriter_config_pb2.RewriterConfig.OFF)

        if mode == 'train' and use_hvd:
            config.intra_op_parallelism_threads = 1  # Avoid pool of Eigen threads
            config.inter_op_parallelism_threads = max(2, (multiprocessing.cpu_count() // max(hvd.size(), 8) - 2))

        return config

    def _get_run_config(self, mode, model_dir, use_xla, use_dali, device_memory_fraction, device_id=0, seed=None, use_hvd=False):

        if mode not in ["train", 'validation', 'benchmark', 'inference']:
            raise ValueError(
                "Unknown mode received: %s (allowed: 'train', 'validation', 'benchmark', 'inference')" % mode
            )

        if seed is not None:
            if use_hvd:
                tf_random_seed = 2 * (seed + hvd.rank())
            else:
                tf_random_seed = 2 * seed
        else:
            tf_random_seed = None


        config = tf.estimator.RunConfig(
            model_dir=model_dir,
            tf_random_seed=tf_random_seed,
            save_summary_steps=100 if mode in ['train', 'validation'] else 1e9,  # disabled in benchmark mode
            save_checkpoints_steps=None,
            save_checkpoints_secs=None,
            session_config=self._get_session_config(
                mode=mode, use_xla=use_xla, use_dali=use_dali, device_memory_fraction=device_memory_fraction, device_id=device_id, use_hvd=use_hvd
            ),
            keep_checkpoint_max=5,
            keep_checkpoint_every_n_hours=1e6,  # disabled
            log_step_count_steps=1e9,
            train_distribute=None,
            device_fn=None,
            protocol=None,
            eval_distribute=None,
            experimental_distribute=None
        )

        if mode == 'train':
            if use_hvd:
                config = config.replace(
                    save_checkpoints_steps=125 if hvd.rank() == 0 else None, keep_checkpoint_every_n_hours=3
                )
            else:
                config = config.replace(save_checkpoints_steps=1000, keep_checkpoint_every_n_hours=3)

        return config

    def _get_estimator(self, mode, run_params, use_xla, use_dali, device_memory_fraction, device_id=0):

        if mode not in ["train", 'validation', 'benchmark', 'inference']:
            raise ValueError(
                "Unknown mode received: %s (allowed: 'train', 'validation', 'benchmark', 'inference')" % mode
            )

        run_config = self._get_run_config(
            mode=mode,
            model_dir=self.run_hparams.model_dir,
            use_xla=use_xla,
            use_dali=use_dali,
            device_memory_fraction=device_memory_fraction,
            device_id=device_id,
            seed=self.run_hparams.seed,
            use_hvd=self.use_horovod_
        )

        return tf.estimator.Estimator(
            model_fn=self._model, model_dir=self.run_hparams.model_dir, config=run_config, params=run_params
        )

    def train(
        self,
        iter_unit,
        num_iter,
        run_iter,
        batch_size,
        warmup_steps=50,
        weight_decay=1e-4,
        lr_init=0.1,
        lr_warmup_epochs=5,
        momentum=0.9,
        log_every_n_steps=1,
        loss_scale=256,
        label_smoothing=0.0,
        mixup=0.0,
        use_cosine_lr=False,
        use_static_loss_scaling=False,
        is_benchmark=False,
        quantize=False,
        symmetric=False,
        quant_delay=0,
        finetune_checkpoint=None,
        use_final_conv=False,
        use_qdq=False
    ):

        if iter_unit not in ["epoch", "batch"]:
            raise ValueError('`iter_unit` value is unknown: %s (allowed: ["epoch", "batch"])' % iter_unit)

        if self.run_hparams.data_dir is None and not is_benchmark:
            raise ValueError('`data_dir` must be specified for training!')

        if self.run_hparams.use_tf_amp or self.run_hparams.dtype == tf.float16:
            if use_static_loss_scaling:
                os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_LOSS_SCALING"] = "0"
            else:
                os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_LOSS_SCALING"] = "1"

        num_devices = 1 if (not self.use_horovod_) else hvd.size()
        global_batch_size = batch_size * num_devices

        # ignore data_dir in benchmark mode
        if is_benchmark:
            self.run_hparams.data_dir = None

        if self.run_hparams.data_dir is not None:
            filenames, num_samples, num_steps, num_epochs, num_decay_steps = runner_utils.parse_tfrecords_dataset(
                data_dir=self.run_hparams.data_dir,
                mode="train",
                iter_unit=iter_unit,
                num_iter=num_iter,
                global_batch_size=global_batch_size,
            )

            steps_per_epoch = num_steps / num_epochs

        else:
            num_epochs = 1
            num_steps = num_iter
            steps_per_epoch = num_steps
            num_decay_steps = num_steps
            num_samples = num_steps * batch_size

        self.global_batch_size = global_batch_size
        self.num_samples = num_samples

        if run_iter == -1:
            run_iter = num_steps
        else:
            run_iter = steps_per_epoch * run_iter if iter_unit == "epoch" else run_iter

        if self.run_hparams.use_dali and self.run_hparams.data_idx_dir is not None:
            idx_filenames = runner_utils.parse_dali_idx_dataset(
                data_idx_dir=self.run_hparams.data_idx_dir, mode="train"
            )

        training_hooks = []

        if self.use_profiler_:
            if (self.use_horovod_ and hvd.rank() == 0) or (not self.use_horovod_):
                from tensorflow_core.python.client import timeline
                timeline_hook = tf.estimator.ProfilerHook(save_steps=5, output_dir='./profiler')
                training_hooks.append(timeline_hook)
        if self.use_performance_:
            if (self.use_horovod_ and hvd.rank() == 0) or (not self.use_horovod_):
                time_hooks = TimeHook()
                training_hooks.append(time_hooks)

        if (self.use_horovod_ and hvd.rank() == 0) or (not self.use_horovod_):
            print('Starting Model Training...')
            print("Training Epochs", num_epochs)
            print("Total Steps", num_steps)
            print("Steps per Epoch", steps_per_epoch)
            print("Decay Steps", num_decay_steps)
            print("Weight Decay Factor", weight_decay)
            print("Init Learning Rate", lr_init)
            print("Momentum", momentum)
            print("Num Devices", num_devices)
            print("Per-Device Batch Size", batch_size)

            if self.hvd_finetune_test:
                print("You are running finetune(resume from a single-card-trained ckpt). So please ignore the epoch num print above and when running iterations, because your global_step is restored from single-card-trained ckpt but steps_per_epoch is calculated by using hvd.size()")

            if is_benchmark:
                self.training_logging_hook = hooks.BenchmarkLoggingHook(
                    global_batch_size=global_batch_size, warmup_steps=warmup_steps, logging_steps=log_every_n_steps
                )
            else:
                self.training_logging_hook = hooks.TrainingLoggingHook(
                    global_batch_size=global_batch_size,
                    num_steps=num_steps,
                    num_samples=num_samples,
                    num_epochs=num_epochs,
                    steps_per_epoch=steps_per_epoch,
                    warmup_steps=warmup_steps,
                    logging_steps=log_every_n_steps
                )
            training_hooks.append(self.training_logging_hook)

        if self.use_horovod_:
            bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
            training_hooks.append(bcast_hook)

        training_hooks.append(hooks.PrefillStagingAreasHook())
        training_hooks.append(hooks.TrainingPartitionHook())

        estimator_params = {
            'batch_size': batch_size,
            'steps_per_epoch': steps_per_epoch,
            'num_devices': num_devices,
            'momentum': momentum,
            'lr_init': lr_init,
            'lr_warmup_epochs': lr_warmup_epochs,
            'weight_decay': weight_decay,
            'loss_scale': loss_scale,
            'apply_loss_scaling': use_static_loss_scaling,
            'label_smoothing': label_smoothing,
            'mixup': mixup,
            'num_decay_steps': num_decay_steps,
            'use_cosine_lr': use_cosine_lr,
            'use_final_conv': use_final_conv,
            'quantize': quantize,
            'use_qdq': use_qdq,
            'symmetric': symmetric,
            'quant_delay': quant_delay,
            'hvd_finetune_test': self.hvd_finetune_test
        }

        if finetune_checkpoint:
            estimator_params['finetune_checkpoint'] = finetune_checkpoint

        image_classifier = self._get_estimator(
            mode='train',
            run_params=estimator_params,
            use_xla=self.run_hparams.use_xla,
            use_dali=self.run_hparams.use_dali,
            device_memory_fraction=self.run_hparams.device_memory_fraction,
            device_id=self.run_hparams.device_id
        )

        def training_data_fn():

            if self.run_hparams.use_dali and self.run_hparams.data_idx_dir is not None:
                if self.use_horovod_ and hvd.rank() == 0:
                    print("Using DALI input... ")
                return mlu_data_utils.get_dali_input_fn(
                    filenames=filenames,
                    idx_filenames=idx_filenames,
                    batch_size=batch_size,
                    height=self.run_hparams.height,
                    width=self.run_hparams.width,
                    training=True,
                    distort_color=self.run_hparams.distort_colors,
                    num_threads=self.run_hparams.num_preprocessing_threads,
                    deterministic=False if self.run_hparams.seed is None else True,
                    device=self.device
                )

            elif self.run_hparams.data_dir is not None:

                return mlu_data_utils.get_tfrecords_input_fn(
                    filenames=filenames,
                    batch_size=batch_size,
                    height=self.run_hparams.height,
                    width=self.run_hparams.width,
                    training=True,
                    distort_color=self.run_hparams.distort_colors,
                    num_threads=self.run_hparams.num_preprocessing_threads,
                    deterministic=False if self.run_hparams.seed is None else True,
                    use_hvd=self.use_horovod_
                )

            else:
                if (self.use_horovod_ and hvd.rank() == 0) or (not self.use_horovod_):
                    print("Using Synthetic Data ...")
                return mlu_data_utils.get_synth_input_fn(
                    batch_size=batch_size,
                    height=self.run_hparams.height,
                    width=self.run_hparams.width,
                    num_channels=self.run_hparams.n_channels,
                    data_format=self.run_hparams.input_format,
                    num_classes=self.run_hparams.n_classes,
                    dtype=self.run_hparams.dtype,
                )

        try:
            current_step = image_classifier.get_variable_value("global_step")
        except ValueError:
            current_step = 0

        run_iter = max(0, min(run_iter, num_steps - current_step))
        print("Current step:", current_step)

        if run_iter > 0:
            try:
                if self.hvd_finetune_test:
                    image_classifier.train(
                        input_fn=training_data_fn,
                        # use max_steps instead of steps, control training steps by using global total step
                        max_steps=num_steps,
                        hooks=training_hooks
                    )
                else:
                    image_classifier.train(
                        input_fn=training_data_fn,
                        steps=run_iter,
                        hooks=training_hooks,
                    )
            except KeyboardInterrupt:
                print("Keyboard interrupt")

        if self.use_performance_ and ((self.use_horovod_ and hvd.rank() == 0) or not self.use_horovod_):
            write_json("summary", global_batch_size, time_hooks.times)

        if (self.use_horovod_ and hvd.rank() == 0) or (not self.use_horovod_):
            if run_iter > 0:
                print('Ending Model Training ...')
                train_throughput = self.training_logging_hook.mean_throughput.value()
                dllogger.log(data={'train_throughput': train_throughput}, step=tuple())
            else:
                print('Model already trained required number of steps. Skipped')

    def evaluate(
        self,
        iter_unit,
        num_iter,
        batch_size,
        warmup_steps=50,
        log_every_n_steps=1,
        is_benchmark=False,
        export_dir=None,
        quantize=False,
        symmetric=False,
        use_qdq=False,
        use_final_conv=False,
    ):

        if iter_unit not in ["epoch", "batch"]:
            raise ValueError('`iter_unit` value is unknown: %s (allowed: ["epoch", "batch"])' % iter_unit)

        if self.run_hparams.data_dir is None and not is_benchmark:
            raise ValueError('`data_dir` must be specified for evaluation!')

        if self.use_horovod_ and hvd.rank() != 0:
            raise RuntimeError('Multi-Device inference is not supported')

        estimator_params = {'quantize': quantize,
                            'symmetric': symmetric,
                            'use_qdq': use_qdq,
                            'use_final_conv': use_final_conv}

        if self.run_hparams.data_dir is not None:
            filenames, num_samples, num_steps, num_epochs, num_decay_steps = runner_utils.parse_tfrecords_dataset(
                data_dir=self.run_hparams.data_dir,
                mode="validation",
                iter_unit=iter_unit,
                num_iter=num_iter,
                global_batch_size=batch_size,
            )

        else:
            num_epochs = 1
            num_decay_steps = -1
            num_steps = num_iter
        self.global_batch_size=batch_size
        self.num_samples=num_samples

        image_classifier = self._get_estimator(
            mode='validation',
            run_params=estimator_params,
            use_xla=self.run_hparams.use_xla,
            use_dali=self.run_hparams.use_dali,
            device_memory_fraction=self.run_hparams.device_memory_fraction,
            device_id=self.run_hparams.device_id
        )

        if self.run_hparams.use_dali and self.run_hparams.data_idx_dir is not None:
            idx_filenames = runner_utils.parse_dali_idx_dataset(
                data_idx_dir=self.run_hparams.data_idx_dir, mode="validation"
            )

        eval_hooks = []

        if (self.use_horovod_ and hvd.rank() == 0) or (not self.use_horovod_):
            self.eval_logging_hook = hooks.BenchmarkLoggingHook(
                global_batch_size=batch_size, warmup_steps=warmup_steps, logging_steps=log_every_n_steps
            )
            eval_hooks.append(self.eval_logging_hook)
            print('Starting Model Evaluation...')
            print("Evaluation Epochs", num_epochs)
            print("Evaluation Steps", num_steps)
            print("Decay Steps", num_decay_steps)
            print("Global Batch Size", batch_size)

        def evaluation_data_fn():

            if self.run_hparams.use_dali and self.run_hparams.data_idx_dir is not None:
                if not self.use_horovod_ or hvd.rank() == 0:
                    print("Using DALI input... ")
                return mlu_data_utils.get_dali_input_fn(
                    filenames=filenames,
                    idx_filenames=idx_filenames,
                    batch_size=batch_size,
                    height=self.run_hparams.height,
                    width=self.run_hparams.width,
                    training=False,
                    distort_color=self.run_hparams.distort_colors,
                    num_threads=self.run_hparams.num_preprocessing_threads,
                    deterministic=False if self.run_hparams.seed is None else True,
                    device=self.device
                )

            elif self.run_hparams.data_dir is not None:
                return mlu_data_utils.get_tfrecords_input_fn(
                    filenames=filenames,
                    batch_size=batch_size,
                    height=self.run_hparams.height,
                    width=self.run_hparams.width,
                    training=False,
                    distort_color=self.run_hparams.distort_colors,
                    num_threads=self.run_hparams.num_preprocessing_threads,
                    deterministic=False if self.run_hparams.seed is None else True,
                    use_hvd=self.use_horovod_
                )

            else:
                print("Using Synthetic Data ...\n")
                return mlu_data_utils.get_synth_input_fn(
                    batch_size=batch_size,
                    height=self.run_hparams.height,
                    width=self.run_hparams.width,
                    num_channels=self.run_hparams.n_channels,
                    data_format=self.run_hparams.input_format,
                    num_classes=self.run_hparams.n_classes,
                    dtype=self.run_hparams.dtype,
                )

        try:
            eval_results = image_classifier.evaluate(
                input_fn=evaluation_data_fn,
                steps=num_steps,
                hooks=eval_hooks,
            )

            eval_throughput = self.eval_logging_hook.mean_throughput.value()
            eval_latencies = np.array(self.eval_logging_hook.latencies) * 1000
            eval_latencies_q = np.quantile(eval_latencies, q=[0.9, 0.95, 0.99])
            eval_latencies_mean = np.mean(eval_latencies)

            dllogger.log(
                data={
                    'top1_accuracy': float(eval_results['top1_accuracy']),
                    'top5_accuracy': float(eval_results['top5_accuracy']),
                    'eval_throughput': eval_throughput,
                    'eval_latency_avg': eval_latencies_mean,
                    'eval_latency_p90': eval_latencies_q[0],
                    'eval_latency_p95': eval_latencies_q[1],
                    'eval_latency_p99': eval_latencies_q[2],
                },
                step=tuple()
            )

            if export_dir is not None:
                dllogger.log(data={'export_dir': export_dir}, step=tuple())
                input_receiver_fn = mlu_data_utils.get_serving_input_receiver_fn(
                    batch_size=None,
                    height=self.run_hparams.height,
                    width=self.run_hparams.width,
                    num_channels=self.run_hparams.n_channels,
                    data_format=self.run_hparams.input_format,
                    dtype=self.run_hparams.dtype
                )

                image_classifier.export_savedmodel(export_dir, input_receiver_fn)

        except KeyboardInterrupt:
            print("Keyboard interrupt")

        print('Model evaluation finished')

    def predict(self, to_predict, quantize=False, symmetric=False, use_qdq=False, use_final_conv=False):

        estimator_params = {'quantize': quantize, 'symmetric': symmetric, 'use_qdq': use_qdq, 'use_final_conv': use_final_conv}

        if to_predict is not None:
            filenames = runner_utils.parse_inference_input(to_predict)

        image_classifier = self._get_estimator(
            mode='inference',
            run_params=estimator_params,
            use_xla=self.run_hparams.use_xla,
            use_dali=self.run_hparams.use_dali,
            device_memory_fraction=self.run_hparams.device_memory_fraction
        )

        inference_hooks = []

        def inference_data_fn():
            return mlu_data_utils.get_inference_input_fn(
                filenames=filenames,
                height=self.run_hparams.height,
                width=self.run_hparams.width,
                num_threads=self.run_hparams.num_preprocessing_threads
            )

        try:
            inference_results = image_classifier.predict(
                input_fn=inference_data_fn, predict_keys=None, hooks=inference_hooks, yield_single_examples=True
            )

            for result in inference_results:
                print(result['classes'], str(result['probabilities'][result['classes']]))

        except KeyboardInterrupt:
            print("Keyboard interrupt")

        print('Ending Inference ...')