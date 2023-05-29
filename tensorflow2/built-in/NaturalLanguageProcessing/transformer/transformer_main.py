# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Train and evaluate the Transformer model.

See README for description of setting the training schedule and evaluating the
BLEU score.
"""

import os
import tempfile

# Import libraries

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import sys
cur_path=os.getcwd()
models_path=cur_path + "/models/"
sys.path.append(models_path)

from official.common import distribute_utils
from official.legacy.transformer import compute_bleu
from official.legacy.transformer import data_pipeline
from official.legacy.transformer import metrics
#import misc
from official.legacy.transformer import misc
from official.legacy.transformer import optimizer
from official.legacy.transformer import transformer
from official.legacy.transformer import translate
from official.legacy.transformer.utils import tokenizer
from official.modeling import performance
from official.utils.flags import core as flags_core
from official.utils.misc import keras_utils

# pylint:disable=logging-format-interpolation

INF = int(1e9)
BLEU_DIR = "bleu"
_SINGLE_SAMPLE = 1

flags.DEFINE_string(name='tf_mlu_thread_mode', default=None,
                    help='Whether and how the MLU device uses its own threadpool.')

flags.DEFINE_integer(name="per_mlu_thread_count", default=0,
                     help='The number of threads to use for MLU. Only valid when '
                     'tf_mlu_thread_mode is not global.')

flags.DEFINE_integer(name="inter_op_parallelism_threads", default=4,
                     help='The number of inter op threads to use for MLU.')

flags.DEFINE_integer(name="intra_op_parallelism_threads", default=4,
                     help='The number of intra op threads to use for MLU.')

flags.DEFINE_integer(name="horovod_fusion_threshold", default=33554432,
                     help='The horovod fusion threshold to use for MLU. Only valid '
                     'when use hvd is True. Default to 32*1024*1024')

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


def translate_and_compute_bleu(model,
                               params,
                               subtokenizer,
                               bleu_source,
                               bleu_ref,
                               distribution_strategy=None):
  """Translate file and report the cased and uncased bleu scores.

  Args:
    model: A Keras model, used to generate the translations.
    params: A dictionary, containing the translation related parameters.
    subtokenizer: A subtokenizer object, used for encoding and decoding source
      and translated lines.
    bleu_source: A file containing source sentences for translation.
    bleu_ref: A file containing the reference for the translated sentences.
    distribution_strategy: A platform distribution strategy, used for TPU based
      translation.

  Returns:
    uncased_score: A float, the case insensitive BLEU score.
    cased_score: A float, the case sensitive BLEU score.
  """
  # Create temporary file to store translation.
  tmp = tempfile.NamedTemporaryFile(delete=False)
  tmp_filename = tmp.name

  translate.translate_file(
      model,
      params,
      subtokenizer,
      bleu_source,
      output_file=tmp_filename,
      print_all_translations=False,
      distribution_strategy=distribution_strategy)

  # Compute uncased and cased bleu scores.
  uncased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, False)
  cased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, True)
  os.remove(tmp_filename)
  return uncased_score, cased_score


def evaluate_and_log_bleu(model,
                          params,
                          bleu_source,
                          bleu_ref,
                          vocab_file,
                          distribution_strategy=None):
  """Calculate and record the BLEU score.

  Args:
    model: A Keras model, used to generate the translations.
    params: A dictionary, containing the translation related parameters.
    bleu_source: A file containing source sentences for translation.
    bleu_ref: A file containing the reference for the translated sentences.
    vocab_file: A file containing the vocabulary for translation.
    distribution_strategy: A platform distribution strategy, used for TPU based
      translation.

  Returns:
    uncased_score: A float, the case insensitive BLEU score.
    cased_score: A float, the case sensitive BLEU score.
  """
  subtokenizer = tokenizer.Subtokenizer(vocab_file)

  uncased_score, cased_score = translate_and_compute_bleu(
      model, params, subtokenizer, bleu_source, bleu_ref, distribution_strategy)

  logging.info("Bleu score (uncased): %s", uncased_score)
  logging.info("Bleu score (cased): %s", cased_score)
  return uncased_score, cased_score


class TransformerTask(object):
  """Main entry of Transformer model."""

  def __init__(self, flags_obj):
    """Init function of TransformerMain.

    Args:
      flags_obj: Object containing parsed flag values, i.e., FLAGS.

    Raises:
      ValueError: if not using static batch for input data on TPU.
    """
    self.flags_obj = flags_obj
    self.predict_model = None

    # Add flag-defined parameters to params object
    num_gpus = flags_core.get_num_gpus(flags_obj)
    self.params = params = misc.get_model_params(flags_obj.param_set, num_gpus)

    params["num_gpus"] = num_gpus
    params["num_mlus"] = flags_obj.num_mlus
    params["use_ctl"] = flags_obj.use_ctl
    params["data_dir"] = flags_obj.data_dir
    params["model_dir"] = flags_obj.model_dir
    params["static_batch"] = flags_obj.static_batch
    params["max_length"] = flags_obj.max_length
    params["decode_batch_size"] = flags_obj.decode_batch_size
    params["decode_max_length"] = flags_obj.decode_max_length
    params["padded_decode"] = flags_obj.padded_decode
    params["max_io_parallelism"] = (
        flags_obj.num_parallel_calls or tf.data.experimental.AUTOTUNE)
    params["use_synthetic_data"] = flags_obj.use_synthetic_data
    params["batch_size"] = flags_obj.batch_size or params["default_batch_size"]
    params["repeat_dataset"] = None
    params["enable_tensorboard"] = flags_obj.enable_tensorboard
    params["enable_metrics_in_training"] = flags_obj.enable_metrics_in_training
    params["steps_between_evals"] = flags_obj.steps_between_evals
    params["enable_checkpointing"] = flags_obj.enable_checkpointing
    params["save_weights_only"] = flags_obj.save_weights_only
    params["use_horovod"] = flags_obj.use_horovod
    params["use_amp"] = flags_obj.use_amp
    params["use_profiler"] = flags_obj.use_profiler
    params["use_performance"] = flags_obj.use_performance
    params["dtype"] = tf.float16 if flags_obj.use_amp else tf.float32
    params["finetune_checkpoint"] = flags_obj.finetune_checkpoint
    params["tf_mlu_thread_mode"] = flags_obj.tf_mlu_thread_mode
    params["tf_gpu_thread_mode"] = flags_obj.tf_gpu_thread_mode
    params["per_mlu_thread_count"] = flags_obj.per_mlu_thread_count
    params["per_gpu_thread_count"] = flags_obj.per_gpu_thread_count
    params["inter_op_parallelism_threads"] = flags_obj.inter_op_parallelism_threads
    params["intra_op_parallelism_threads"] = flags_obj.intra_op_parallelism_threads
    params["horovod_fusion_threshold"] = flags_obj.horovod_fusion_threshold

    if  flags_obj.enable_xla:
      # The input shape must be static when running with XLA.
      params['static_batch'] = True
      os.environ['TF_XLA_FLAGS'] = (os.environ.get("TF_XLA_FLAGS", "") +
          " --tf_xla_auto_jit=2 --tf_xla_enable_lazy_compilation=false --tf_xla_async_io_level=1 ")
      os.environ['XLA_MLU_DISABLE_BITCAST_OPT'] = 'true'

    if params["use_horovod"]:
       os.environ['HOROVOD_FUSION_THRESHOLD'] = str(flags_obj.horovod_fusion_threshold)
       logging.info('HOROVOD_FUSION_THRESHOLD: %s', os.environ['HOROVOD_FUSION_THRESHOLD'])
       import horovod.tensorflow.keras as hvd
       global hvd
       hvd.init()
    if params["use_performance"]:
        from record_time import TimeHistoryRecord, write_json
        global TimeHistoryRecord
        global write_json
    if params["use_performance"] and params["use_profiler"]:
      raise ValueError("You can only set use_profiler or use_performance, not at the same time, otherwise the e2e time will be worse")
    if tf.config.list_physical_devices('MLU'):
        if params["tf_mlu_thread_mode"]:
            set_thread_mode_and_count(
                    mlu_thread_mode=flags_obj.tf_mlu_thread_mode,
                    per_mlu_thread_count=flags_obj.per_mlu_thread_count,
                    interop_threads=flags_obj.inter_op_parallelism_threads,
                    intraop_threads=flags_obj.intra_op_parallelism_threads)
    elif tf.config.list_physical_devices('GPU'):
        if params["tf_gpu_thread_mode"]:
            keras_utils.set_gpu_thread_mode_and_count(
                    per_gpu_thread_count=flags_obj.per_gpu_thread_count,
                    gpu_thread_mode=flags_obj.tf_gpu_thread_mode,
                    num_gpus=flags_obj.num_gpus,
                    datasets_num_private_threads=flags_obj.datasets_num_private_threads)

    self.distribution_strategy = distribute_utils.get_distribution_strategy(
        distribution_strategy=flags_obj.distribution_strategy,
        num_gpus=num_gpus,
        num_mlus=flags_obj.num_mlus,
        all_reduce_alg=flags_obj.all_reduce_alg,
        num_packs=flags_obj.num_packs,
        tpu_address=flags_obj.tpu or "")
    if self.use_tpu:
      params["num_replicas"] = self.distribution_strategy.num_replicas_in_sync
    elif num_gpus > 0:
      logging.info("Running transformer with num_gpus = %d", num_gpus)
    elif flags_obj.num_mlus > 0:
      logging.info("Running transformer with num_mlus = %d", flags_obj.num_mlus)

    if self.distribution_strategy:
      logging.info("For training, using distribution strategy: %s",
                   self.distribution_strategy)
    else:
      logging.info("Not using any distribution strategy.")

    performance.set_mixed_precision_policy(params["dtype"])

  @property
  def use_tpu(self):
    if self.distribution_strategy:
      return isinstance(self.distribution_strategy, tf.distribute.TPUStrategy)
    return False

  def train(self):
    """Trains the model."""
    params = self.params
    flags_obj = self.flags_obj

    if params['use_horovod']:
      if params['num_gpus'] > 0 and params['num_mlus'] <= 0:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
          tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
      elif params['num_gpus'] <= 0 and params['num_mlus'] > 0:
        mlus = tf.config.experimental.list_physical_devices('MLU')
        if mlus:
          tf.config.experimental.set_visible_devices(mlus[hvd.local_rank()], 'MLU')
      else:
        raise ValueError("Only one of mlu and gpu can be greater than 0")

    # Sets config options.

    _ensure_dir(flags_obj.model_dir)
    with distribute_utils.get_strategy_scope(self.distribution_strategy):
      model = transformer.create_model(params, is_train=True)
      opt = self._create_optimizer()

      current_step = 0
      checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)
      if params["finetune_checkpoint"]:
        latest_checkpoint = tf.train.latest_checkpoint(flags_obj.finetune_checkpoint)
      else:
        latest_checkpoint = tf.train.latest_checkpoint(flags_obj.model_dir)
      if latest_checkpoint:
        model.load_weights(latest_checkpoint).expect_partial()
        logging.info("Loaded checkpoint %s", latest_checkpoint)
        current_step = opt.iterations.numpy()
        flags_obj.train_steps += current_step

      if params["use_ctl"]:
        train_loss_metric = tf.keras.metrics.Mean(
            "training_loss", dtype=tf.float32)
        if params["enable_tensorboard"]:
          summary_writer = tf.summary.create_file_writer(
              os.path.join(flags_obj.model_dir, "summary"))
        else:
          summary_writer = tf.summary.create_noop_writer()
        train_metrics = [train_loss_metric]
        if params["enable_metrics_in_training"]:
          train_metrics = train_metrics + model.metrics
      else:
        model.compile(opt)

    model.summary()

    if self.use_tpu:
      # Different from experimental_distribute_dataset,
      # distribute_datasets_from_function requires
      # per-replica/local batch size.
      params["batch_size"] /= self.distribution_strategy.num_replicas_in_sync
      train_ds = (
          self.distribution_strategy.distribute_datasets_from_function(
              lambda ctx: data_pipeline.train_input_fn(params, ctx)))
    else:
      if params['use_horovod']:
        # use this to do training data partition when use horovod
        ctx = tf.distribute.InputContext(num_input_pipelines=hvd.size(), input_pipeline_id=hvd.rank(), num_replicas_in_sync=hvd.size())
      else:
        ctx = None
      train_ds = data_pipeline.train_input_fn(params, ctx)
      map_data_fn = data_pipeline.map_data_for_transformer_fn
      train_ds = train_ds.map(
          map_data_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if params["use_ctl"]:
      train_ds_iterator = iter(train_ds)

    callbacks = self._create_callbacks(flags_obj.model_dir, params)

    # Only TimeHistory callback is supported for CTL
    if params["use_ctl"]:
      callbacks = [cb for cb in callbacks
                   if isinstance(cb, keras_utils.TimeHistory)]

    @tf.function
    def train_steps(iterator, steps):
      """Training steps function for TPU runs.

      Args:
        iterator: The input iterator of the training dataset.
        steps: An integer, the number of training steps.

      Returns:
        A float, the loss value.
      """

      def _step_fn(inputs):
        """Per-replica step function."""
        inputs, targets = inputs
        with tf.GradientTape() as tape:
          logits = model([inputs, targets], training=True)
          loss = metrics.transformer_loss(logits, targets,
                                          params["label_smoothing"],
                                          params["vocab_size"])
          # Scales the loss, which results in using the average loss across all
          # of the replicas for backprop.
          scaled_loss = loss / self.distribution_strategy.num_replicas_in_sync

        # De-dupes variables due to keras tracking issues.
        tvars = list({id(v): v for v in model.trainable_variables}.values())
        grads = tape.gradient(scaled_loss, tvars)
        opt.apply_gradients(zip(grads, tvars))
        # For reporting, the metric takes the mean of losses.
        train_loss_metric.update_state(loss)

      for _ in tf.range(steps):
        train_loss_metric.reset_states()
        self.distribution_strategy.run(
            _step_fn, args=(next(iterator),))

    cased_score, uncased_score = None, None
    cased_score_history, uncased_score_history = [], []
    while current_step < flags_obj.train_steps:
      remaining_steps = flags_obj.train_steps - current_step
      train_steps_per_eval = (
          remaining_steps if remaining_steps < flags_obj.steps_between_evals
          else flags_obj.steps_between_evals)
      current_iteration = current_step // flags_obj.steps_between_evals

      logging.info(
          "Start train iteration at global step:{}".format(current_step))
      history = None
      if params["use_ctl"]:
        if not self.use_tpu:
          raise NotImplementedError(
              "Custom training loop on GPUs is not implemented.")

        # Runs training steps.
        with summary_writer.as_default():
          for cb in callbacks:
            cb.on_epoch_begin(current_iteration)
            cb.on_batch_begin(0)

          train_steps(
              train_ds_iterator,
              tf.convert_to_tensor(train_steps_per_eval, dtype=tf.int32))
          current_step += train_steps_per_eval
          train_loss = train_loss_metric.result().numpy().astype(float)
          logging.info("Train Step: %d/%d / loss = %s", current_step,
                       flags_obj.train_steps, train_loss)

          for cb in callbacks:
            cb.on_batch_end(train_steps_per_eval - 1)
            cb.on_epoch_end(current_iteration)

          if params["enable_tensorboard"]:
            for metric_obj in train_metrics:
              tf.summary.scalar(metric_obj.name, metric_obj.result(),
                                current_step)
              summary_writer.flush()

        for cb in callbacks:
          cb.on_train_end()

        if flags_obj.enable_checkpointing:
          # avoid check-pointing when running for benchmarking.
          checkpoint_name = checkpoint.save(
              os.path.join(flags_obj.model_dir,
                           "ctl_step_{}.ckpt".format(current_step)))
          logging.info("Saved checkpoint to %s", checkpoint_name)
      else:
        if self.use_tpu:
          raise NotImplementedError(
              "Keras model.fit on TPUs is not implemented.")
        if params['use_performance'] and ((params['use_horovod'] and hvd.rank() == 0) or not params['use_horovod']):
          time_callback = TimeHistoryRecord()
          callbacks.append(time_callback)

        vbs = 2 if flags_obj.enable_time_history else 1
        if params['use_horovod']:
          vbs = 1 if hvd.rank() == 0 else 0
        history = model.fit(
            train_ds,
            initial_epoch=current_iteration,
            epochs=current_iteration + 1,
            steps_per_epoch=train_steps_per_eval,
            callbacks=callbacks,
            # If TimeHistory is enabled, progress bar would be messy. Increase
            # the verbose level to get rid of it.
            verbose=vbs)
        current_step += train_steps_per_eval
        logging.info("Train history: {}".format(history.history))

      logging.info("End train iteration at global step:{}".format(current_step))

      if (flags_obj.bleu_source and flags_obj.bleu_ref):
        uncased_score, cased_score = self.eval()
        cased_score_history.append([current_iteration + 1, cased_score])
        uncased_score_history.append([current_iteration + 1, uncased_score])

    if params['use_performance']:
      global_batch_size = params["batch_size"] if not params['use_horovod'] else params["batch_size"] * hvd.size()
      if (params['use_horovod'] and hvd.rank() == 0) or not params['use_horovod']:
        write_json("summary", global_batch_size, time_callback.times)

    stats = ({
        "loss": train_loss
    } if history is None else {})
    misc.update_stats(history, stats, callbacks)
    if uncased_score and cased_score:
      stats["bleu_uncased"] = uncased_score
      stats["bleu_cased"] = cased_score
      stats["bleu_uncased_history"] = uncased_score_history
      stats["bleu_cased_history"] = cased_score_history
    return stats

  def eval(self):
    """Evaluates the model."""
    distribution_strategy = self.distribution_strategy if self.use_tpu else None

    # We only want to create the model under DS scope for TPU case.
    # When 'distribution_strategy' is None, a no-op DummyContextManager will
    # be used.
    with distribute_utils.get_strategy_scope(distribution_strategy):
      if not self.predict_model:
        self.predict_model = transformer.create_model(self.params, False)
      self._load_weights_if_possible(
          self.predict_model,
          tf.train.latest_checkpoint(self.flags_obj.model_dir))
      self.predict_model.summary()
    return evaluate_and_log_bleu(
        self.predict_model, self.params, self.flags_obj.bleu_source,
        self.flags_obj.bleu_ref, self.flags_obj.vocab_file,
        distribution_strategy)

  def predict(self):
    """Predicts result from the model."""
    params = self.params
    flags_obj = self.flags_obj

    with tf.name_scope("model"):
      model = transformer.create_model(params, is_train=False)
      self._load_weights_if_possible(
          model, tf.train.latest_checkpoint(self.flags_obj.model_dir))
      model.summary()
    subtokenizer = tokenizer.Subtokenizer(flags_obj.vocab_file)

    ds = data_pipeline.eval_input_fn(params)
    ds = ds.map(lambda x, y: x).take(_SINGLE_SAMPLE)
    ret = model.predict(ds)
    val_outputs, _ = ret
    length = len(val_outputs)
    for i in range(length):
      translate.translate_from_input(val_outputs[i], subtokenizer)

  def _create_callbacks(self, cur_log_dir, params):
    """Creates a list of callbacks."""
    callbacks = misc.get_callbacks()
    if params["enable_checkpointing"]:
      ckpt_full_path = os.path.join(cur_log_dir, "cp-{epoch:04d}.ckpt")
      if ((params['use_horovod'] and hvd.rank() == 0) or (not params['use_horovod'])):
        callbacks.append(
          tf.keras.callbacks.ModelCheckpoint(
            ckpt_full_path, save_weights_only=params["save_weights_only"]))
    if params["use_horovod"]:
      callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
    if params["use_profiler"] and ((params['use_horovod'] and hvd.rank() == 0) or (not params['use_horovod'])):
      callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=params["model_dir"], profile_batch=2))
    return callbacks

  def _load_weights_if_possible(self, model, init_weight_path=None):
    """Loads model weights when it is provided."""
    if init_weight_path:
      logging.info("Load weights: {}".format(init_weight_path))
      if self.use_tpu:
        checkpoint = tf.train.Checkpoint(
            model=model, optimizer=self._create_optimizer())
        checkpoint.restore(init_weight_path)
      else:
        model.load_weights(init_weight_path)
    else:
      logging.info("Weights not loaded from path:{}".format(init_weight_path))

  def _create_optimizer(self):
    """Creates optimizer."""
    params = self.params
    lr_schedule = optimizer.LearningRateSchedule(
        params["learning_rate"], params["hidden_size"],
        params["learning_rate_warmup_steps"])
    opt = tf.keras.optimizers.legacy.Adam(
        lr_schedule,
        params["optimizer_adam_beta1"],
        params["optimizer_adam_beta2"],
        epsilon=params["optimizer_adam_epsilon"])

    if params['use_horovod']:
      with tf.keras.utils.CustomObjectScope({lr_schedule.__class__.__name__: lr_schedule}):
        opt = hvd.DistributedOptimizer(opt, num_groups=15)

    opt = performance.configure_optimizer(
        opt,
        use_float16=params["dtype"] == tf.float16,
        loss_scale=flags_core.get_loss_scale(
            self.flags_obj, default_for_fp16="dynamic"))

    return opt


def _ensure_dir(log_dir):
  """Makes log dir if not existed."""
  if not tf.io.gfile.exists(log_dir):
    tf.io.gfile.makedirs(log_dir)


def main(_):
  flags_obj = flags.FLAGS
  if flags_obj.enable_mlir_bridge:
    tf.config.experimental.enable_mlir_bridge()
  task = TransformerTask(flags_obj)

  if flags_obj.mode == "train":
    task.train()
  elif flags_obj.mode == "predict":
    task.predict()
  elif flags_obj.mode == "eval":
    task.eval()
  else:
    raise ValueError("Invalid mode {}".format(flags_obj.mode))


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  misc.define_transformer_flags()
  app.run(main)
