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
"""Train and evaluate the Transformer model.

See README for description of setting the training schedule and evaluating the
BLEU score.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from nlp.modeling import performance
from nlp.transformer import compute_bleu
from nlp.transformer import data_pipeline
from nlp.transformer import metrics
import misc
from nlp.transformer import optimizer
from nlp.transformer import transformer
from nlp.transformer import translate
from nlp.transformer.utils import tokenizer
from nlp.utils.flags import core as flags_core
from nlp.utils.logs import logger
from nlp.utils.misc import distribution_utils
from nlp.utils.misc import keras_utils

from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf import config_pb2


INF = int(1e9)
BLEU_DIR = "bleu"
_SINGLE_SAMPLE = 1


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
    num_mlus = flags_obj.num_mlus
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
    params["num_parallel_calls"] = (
        flags_obj.num_parallel_calls or tf.data.experimental.AUTOTUNE)

    params["use_synthetic_data"] = flags_obj.use_synthetic_data
    params["batch_size"] = flags_obj.batch_size or params["default_batch_size"]
    params["repeat_dataset"] = None
    params["dtype"] = flags_core.get_tf_dtype(flags_obj)
    params["enable_tensorboard"] = flags_obj.enable_tensorboard
    params["enable_metrics_in_training"] = flags_obj.enable_metrics_in_training
    params["steps_between_evals"] = flags_obj.steps_between_evals
    params["enable_checkpointing"] = flags_obj.enable_checkpointing
    params["save_ckpt_steps"] = flags_obj.save_ckpt_steps
    params["use_horovod"] = flags_obj.use_horovod
    params["use_profiler"] = flags_obj.use_profiler
    params["use_performance"] = flags_obj.use_performance

    if params["use_horovod"]:
       import horovod.tensorflow.keras as hvd
       global hvd
       hvd.init()
    if params["use_performance"]:
        from record_time import TimeHistoryRecord, write_json
        global TimeHistoryRecord
        global write_json
    if params["use_performance"] and params["use_profiler"]:
      raise ValueError("You can only set use_profiler or use_performance, not at the same time, otherwise the e2e time will be worse")

    self.distribution_strategy = distribution_utils.get_distribution_strategy(
        distribution_strategy=flags_obj.distribution_strategy,
        num_gpus=num_gpus,
        all_reduce_alg=flags_obj.all_reduce_alg,
        num_packs=flags_obj.num_packs,
        tpu_address=flags_obj.tpu or "",
        num_mlus=num_mlus)

    if self.use_tpu:
      params["num_replicas"] = self.distribution_strategy.num_replicas_in_sync
      if not params["static_batch"]:
        raise ValueError("TPU requires static batch for input data.")
    else:
      if not params['use_horovod']:
        logging.info("Running transformer with num_mlus = %d", num_mlus)
      elif hvd.rank() == 0:
        logging.info("Running transformer with num_mlus = %d", hvd.size())

    if self.distribution_strategy:
      if not params["use_horovod"]:
        logging.info("For training, using distribution strategy: %s",
                   self.distribution_strategy)
    else:
      logging.info("Not using any distribution strategy.")

    performance.set_mixed_precision_policy(
        params["dtype"],
        flags_core.get_loss_scale(flags_obj, default_for_fp16="dynamic"))

  @property
  def use_tpu(self):
    if self.distribution_strategy:
      return isinstance(self.distribution_strategy,
                        tf.distribute.experimental.TPUStrategy)
    return False

  def train(self):
    """Trains the model."""

    params = self.params

    if params["use_horovod"]:
      hvd.init()

    flags_obj = self.flags_obj

    # Use quant ops
    global_batch_size = flags_obj.batch_size if not params['use_horovod'] else flags_obj.batch_size * hvd.size()
    if params['num_mlus'] > 0:
      rewrite_config = tf.ConfigProto()

      if params['use_horovod']:
        rewrite_config.mlu_options.visible_device_list = str(hvd.local_rank())
    else:
      rewrite_config = tf.ConfigProto()
      if params['use_horovod']:
        rewrite_config.gpu_options.visible_device_list = str(hvd.local_rank())

    rewrite_config.inter_op_parallelism_threads=flags_obj.inter_op_threads
    rewrite_config.intra_op_parallelism_threads=flags_obj.intra_op_threads

    sess = tf.Session(config=rewrite_config)
    tf.compat.v1.keras.backend.set_session(sess)

    run_options = None
    run_metadata = None
    if params['use_profiler'] and ((params['use_horovod'] and hvd.rank() == 0) or (not params['use_horovod'])):
      from tensorflow_core.python.client import timeline
      run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
      run_metadata = tf.compat.v1.RunMetadata()
    _ensure_dir(flags_obj.model_dir)
    with distribution_utils.get_strategy_scope(self.distribution_strategy):
      model = transformer.create_model(params, is_train=True)
      opt = self._create_optimizer()

      if params["use_ctl"]:
        train_loss_metric = tf.keras.metrics.Mean(
            "training_loss", dtype=tf.float32)
        if params["enable_tensorboard"]:
          summary_writer = tf.compat.v2.summary.create_file_writer(
              flags_obj.model_dir)
        else:
          summary_writer = tf.compat.v2.summary.create_noop_writer()
        train_metrics = [train_loss_metric]
        if params["enable_metrics_in_training"]:
          train_metrics = train_metrics + model.metrics
      else:
        if (params['use_horovod'] and hvd.rank() == 0) or (not params['use_horovod']):
          model.compile(opt, options=run_options, run_metadata=run_metadata)
        else:
          model.compile(opt)

    current_step = 0
    checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)
    latest_checkpoint = tf.train.latest_checkpoint(flags_obj.model_dir)
    if latest_checkpoint:
      logging.info("Loaded checkpoint %s", latest_checkpoint)
      #current_step = opt.iterations.numpy()
      model.load_weights(latest_checkpoint).expect_partial()
      import re
      reader = tf.compat.v1.train.NewCheckpointReader(latest_checkpoint)
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in var_to_shape_map:
        if re.search('iter', key):
          current_step = reader.get_tensor(key)

      if current_step > flags_obj.train_steps:
        raise ValueError("current_step is greater than train_steps, "
                         "current_step={}, train_steps={}, "
                         "please check the train_step!".format(current_step,
                         flags_obj.train_steps))

    model.summary()

    if self.use_tpu:
      # Different from experimental_distribute_dataset,
      # experimental_distribute_datasets_from_function requires
      # per-replica/local batch size.
      params["batch_size"] /= self.distribution_strategy.num_replicas_in_sync
      train_ds = (
          self.distribution_strategy
          .experimental_distribute_datasets_from_function(
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
          map_data_fn, num_parallel_calls=params["num_parallel_calls"])
    if params["use_ctl"]:
      train_ds_iterator = iter(train_ds)

    callbacks = self._create_callbacks(flags_obj.model_dir, 0, params)

    # TODO(b/139418525): Refactor the custom training loop logic.
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
          train_steps(
              train_ds_iterator,
              tf.convert_to_tensor(train_steps_per_eval, dtype=tf.int32))
          current_step += train_steps_per_eval
          train_loss = train_loss_metric.result().numpy().astype(float)
          logging.info("Train Step: %d/%d / loss = %s", current_step,
                       flags_obj.train_steps, train_loss)

          if params["enable_tensorboard"]:
            for metric_obj in train_metrics:

              tf.compat.v2.summary.scalar(metric_obj.name, metric_obj.result(),
                                          current_step)

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


        # If TimeHistory is enabled, progress bar would be messy. Increase
        # the verbose level to get rid of it.
        vbs = 2 if flags_obj.enable_time_history else 1
        if params['use_horovod']:
          vbs = 1 if hvd.rank() == 0 else 0
        history = model.fit(
            train_ds,
            initial_epoch=current_iteration,
            epochs=current_iteration + 1,
            steps_per_epoch=train_steps_per_eval,
            callbacks=callbacks,
            verbose=vbs)
        current_step += train_steps_per_eval
        logging.info("Train history: {}".format(history.history))

      logging.info("End train iteration at global step:{}".format(current_step))

      if (flags_obj.bleu_source and flags_obj.bleu_ref):
        if ((params['use_horovod'] and hvd.rank() == 0) or (not params['use_horovod'])):
          uncased_score, cased_score = self.eval()
          cased_score_history.append([current_iteration + 1, cased_score])
          uncased_score_history.append([current_iteration + 1, uncased_score])

    if params['use_profiler'] and ((params['use_horovod'] and hvd.rank() == 0) or not params['use_horovod']):
      tl = timeline.Timeline(run_metadata.step_stats)
      ctf = tl.generate_chrome_trace_format()
      if not tf.io.gfile.exists("profiler"):
        tf.io.gfile.mkdir("profiler")
      with open("profiler/timeline.json", 'w') as f:
        f.write(ctf)
    if params['use_performance'] and ((params['use_horovod'] and hvd.rank() == 0) or not params['use_horovod']):
      write_json("summary", global_batch_size, time_callback.times)

    stats = ({
        "loss": train_loss
    } if history is None else misc.build_stats(history, callbacks))
    if uncased_score and cased_score:
      stats["bleu_uncased"] = uncased_score
      stats["bleu_cased"] = cased_score
      stats["bleu_uncased_history"] = uncased_score_history
      stats["bleu_cased_history"] = cased_score_history

    # Clear the session explicitly to avoid session delete error
    tf.compat.v1.keras.backend.clear_session()

    return stats

  def eval(self):
    """Evaluates the model."""
    distribution_strategy = self.distribution_strategy if self.use_tpu else None

    # We only want to create the model under DS scope for TPU case.
    # When 'distribution_strategy' is None, a no-op DummyContextManager will
    # be used.
    with distribution_utils.get_strategy_scope(distribution_strategy):
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

  def _create_callbacks(self, cur_log_dir, init_steps, params):
    """Creates a list of callbacks."""
    sfunc = optimizer.LearningRateFn(params["learning_rate"],
                                     params["hidden_size"],
                                     params["learning_rate_warmup_steps"])
    scheduler_callback = optimizer.LearningRateScheduler(sfunc, init_steps)
    if (params['use_horovod'] and hvd.rank() == 0) or not params['use_horovod']:
      callbacks = misc.get_callbacks(params["steps_between_evals"])
    else:
      callbacks = []
    callbacks.append(scheduler_callback)
    if params["enable_checkpointing"]:
      ckpt_full_path = os.path.join(cur_log_dir, "cp-{epoch:04d}.ckpt")
      if ((params['use_horovod'] and hvd.rank() == 0) or (not params['use_horovod'])):
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                ckpt_full_path, save_weights_only=True, save_freq=params["save_ckpt_steps"]))
    if params["use_horovod"]:
      callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
    return callbacks

  def _load_weights_if_possible(self, model, init_weight_path=None):
    """Loads model weights when it is provided."""
    if init_weight_path:
      logging.info("Load weights: {}".format(init_weight_path))
      # TODO(b/139414977): Having the same variable restoring method for both
      # TPU and GPU.
      if self.use_tpu:
        checkpoint = tf.train.Checkpoint(
            model=model, optimizer=self._create_optimizer())
        checkpoint.restore(init_weight_path)
      else:
        model.load_weights(init_weight_path).expect_partial()
    else:
      logging.info("Weights not loaded from path:{}".format(init_weight_path))

  def _create_optimizer(self):
    """Creates optimizer."""
    params = self.params
    base_lr = params["learning_rate"]
    if params["use_horovod"]:
      base_lr *= hvd.size()
    lr_schedule = optimizer.LearningRateSchedule(
        base_lr, params["hidden_size"],
        params["learning_rate_warmup_steps"])
    opt = tf.keras.optimizers.Adam(
        lr_schedule if self.use_tpu else params["learning_rate"],
        params["optimizer_adam_beta1"],
        params["optimizer_adam_beta2"],
        epsilon=params["optimizer_adam_epsilon"])

    if params['use_horovod']:
      opt = hvd.DistributedOptimizer(opt)

    # TODO(zhaoyijia) replace native tensorflow optimizer
    # opt = performance.configure_optimizer(
    #     opt,
    #     use_float16=params["dtype"] == tf.float16,
    #     use_graph_rewrite=self.flags_obj.fp16_implementation == "graph_rewrite",
    #     loss_scale=flags_core.get_loss_scale(
    #         self.flags_obj, default_for_fp16="dynamic"))

    logging.info("====== Slot Variables Names ======")
    logging.info(opt.get_slot_names())

    return opt


def _ensure_dir(log_dir):
  """Makes log dir if not existed."""
  if not tf.io.gfile.exists(log_dir):
    tf.io.gfile.makedirs(log_dir)


def main(_):
  flags_obj = flags.FLAGS
  with logger.benchmark_context(flags_obj):
    task = TransformerTask(flags_obj)

    # Execute flag override logic for better model performance
    if flags_obj.tf_gpu_thread_mode:
      keras_utils.set_gpu_thread_mode_and_count(
          per_gpu_thread_count=flags_obj.per_gpu_thread_count,
          gpu_thread_mode=flags_obj.tf_gpu_thread_mode,
          num_gpus=flags_obj.num_gpus,
          datasets_num_private_threads=flags_obj.datasets_num_private_threads)

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
