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

from models import orbit
import tensorflow as tf
import time
import os

from models.orbit.utils import loop_fns
from models.modeling import performance
from models.utils.flags import core as flags_core
from models.resnet import grad_utils
from models.resnet import common
from models.resnet import imagenet_preprocessing
from models.resnet import resnet_model


# copy from orbit/standard_runner.py
def _create_train_loop_fn(train_step_fn, use_tf_while_loop, use_tf_function):
  """Creates a training loop from the given step function and options."""
  if use_tf_while_loop:
    loop_fn = loop_fns.create_tf_while_loop_fn(train_step_fn)
    loop_fn = tf.function(loop_fn)
  else:
    if use_tf_function:
      train_step_fn = tf.function(train_step_fn)
    loop_fn = loop_fns.create_loop_fn(train_step_fn)
  return loop_fn

class ResnetRunnable(orbit.StandardTrainer, orbit.StandardEvaluator):
  """Implements the training and evaluation APIs for Resnet model."""

  def __init__(self, flags_obj, time_callback, epoch_steps, use_performance, use_profiler, hvd, profiler_dir, host_tracer_level=0, device_tracer_level=0, base_learning_rate=0.1):
    self.strategy = tf.distribute.get_strategy()
    self.flags_obj = flags_obj
    self.dtype = flags_core.get_tf_dtype(flags_obj)
    self.time_callback = time_callback
    self.hvd = hvd
    self.time_list = []
    self.use_performance = use_performance
    self.use_profiler = use_profiler
    self.log_dir = None
    self.host_tracer_level = host_tracer_level
    self.device_tracer_level = device_tracer_level
    self.base_learning_rate=base_learning_rate
    if self.use_profiler:
      import tensorflow.profiler.experimental as profiler
      global profiler
      self.log_dir = os.path.join(profiler_dir, 'train')

    # Input pipeline related
    self.total_batch_size = flags_obj.batch_size if hvd is None else flags_obj.batch_size * hvd.size()
    if self.total_batch_size % self.strategy.num_replicas_in_sync != 0:
      raise ValueError(
          'Batch size must be divisible by number of replicas : {}'.format(
              self.strategy.num_replicas_in_sync))

    # As auto rebatching is not supported in
    # `distribute_datasets_from_function()` API, which is
    # required when cloning dataset to multiple workers in eager mode,
    # we use per-replica batch size.
    self.batch_size = int(self.flags_obj.batch_size / self.strategy.num_replicas_in_sync)

    if self.flags_obj.use_synthetic_data:
      self.input_fn = common.get_synth_input_fn(
          height=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
          width=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
          num_channels=imagenet_preprocessing.NUM_CHANNELS,
          num_classes=imagenet_preprocessing.NUM_CLASSES,
          dtype=self.dtype,
          drop_remainder=True)
    else:
      self.input_fn = imagenet_preprocessing.input_fn

    self.model = resnet_model.resnet50(
        num_classes=imagenet_preprocessing.NUM_CLASSES,
        use_l2_regularizer=not flags_obj.single_l2_loss_op)
    lr_schedule = common.PiecewiseConstantDecayWithWarmup(
        batch_size=self.total_batch_size,
        epoch_size=imagenet_preprocessing.NUM_IMAGES['train'],
        warmup_epochs=common.LR_SCHEDULE[0][1],
        boundaries=list(p[1] for p in common.LR_SCHEDULE[1:]),
        multipliers=list(p[0] for p in common.LR_SCHEDULE),
        compute_lr_on_cpu=True, base_learning_rate=self.base_learning_rate)

    self.optimizer = common.get_optimizer(lr_schedule)
    # Make sure iterations variable is created inside scope.
    self.global_step = self.optimizer.iterations

    self.optimizer = performance.configure_optimizer(
        self.optimizer,
        use_float16=self.dtype == tf.float16,
        loss_scale=flags_core.get_loss_scale(flags_obj, default_for_fp16=128))

    self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'train_accuracy', dtype=tf.float32)
    self.test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'test_accuracy', dtype=tf.float32)
    self.test_accuracy_5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(
      k=5, name='test_accuracy_5')

    self.checkpoint = tf.train.Checkpoint(
        model=self.model, optimizer=self.optimizer)

    # Handling epochs.
    self.epoch_steps = epoch_steps
    self.epoch_helper = orbit.utils.EpochHelper(epoch_steps, self.global_step)
    train_dataset = orbit.utils.make_distributed_dataset(
        self.strategy,
        self.input_fn,
        is_training=True,
        data_dir=self.flags_obj.data_dir,
        batch_size=self.batch_size,
        parse_record_fn=imagenet_preprocessing.parse_record,
        datasets_num_private_threads=self.flags_obj
        .datasets_num_private_threads,
        dtype=self.dtype,
        drop_remainder=True,
        hvd=self.hvd)
    orbit.StandardTrainer.__init__(
        self,
        train_dataset,
        options=orbit.StandardTrainerOptions(
            use_tf_while_loop=flags_obj.use_tf_while_loop,
            use_tf_function=flags_obj.use_tf_function))
    if not flags_obj.skip_eval:
      eval_dataset = orbit.utils.make_distributed_dataset(
          self.strategy,
          self.input_fn,
          is_training=False,
          data_dir=self.flags_obj.data_dir,
          batch_size=self.batch_size,
          parse_record_fn=imagenet_preprocessing.parse_record,
          dtype=self.dtype)
      orbit.StandardEvaluator.__init__(
          self,
          eval_dataset,
          options=orbit.StandardEvaluatorOptions(
              use_tf_function=flags_obj.use_tf_function))

  def train_loop_begin(self):
    """See base class."""
    # Reset all metrics
    self.train_loss.reset_states()
    self.train_accuracy.reset_states()

    if self.use_performance and ((self.hvd is not None and self.hvd.rank() == 0) or self.hvd is None):
      self.start = time.time()
    if self.use_profiler and ((self.hvd is not None and self.hvd.rank() == 0) or self.hvd is None):
      options = profiler.ProfilerOptions(host_tracer_level=self.host_tracer_level,
                                         device_tracer_level=self.device_tracer_level)
      profiler.start(self.log_dir, options=options)
    if (self.hvd is not None and self.hvd.rank() == 0) or self.hvd is None:
      self._epoch_begin()
      self.time_callback.on_batch_begin(self.epoch_helper.batch_index)

  def train_step(self, iterator):
    """See base class."""

    def step_fn(inputs):
      """Function to run on the device."""
      images, labels = inputs
      images = tf.cast(images, self.dtype)
      with tf.GradientTape() as tape:
        logits = self.model(images, training=True)

        prediction_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits)
        loss = tf.reduce_sum(prediction_loss) * (1.0 /
                                                 self.total_batch_size)
        num_replicas = self.strategy.num_replicas_in_sync if self.hvd is None else self.hvd.size()
        l2_weight_decay = 1e-4
        if self.flags_obj.single_l2_loss_op:
          l2_loss = l2_weight_decay * 2 * tf.add_n([
              tf.nn.l2_loss(v)
              for v in self.model.trainable_variables
              if 'bn' not in v.name
          ])

          loss += (l2_loss / num_replicas)
        else:
          loss += (tf.reduce_sum(self.model.losses) / num_replicas)
      if self.hvd is not None:
        tape = self.hvd.DistributedGradientTape(tape, op=self.hvd.Sum, num_groups=15)
      grad_utils.minimize_using_explicit_allreduce(
          tape, self.optimizer, loss, self.model.trainable_variables)
      self.train_loss.update_state(loss)
      self.train_accuracy.update_state(labels, logits)
    self.strategy.run(step_fn, args=(next(iterator),))

  def train_loop_end(self):
    """See base class."""
    metrics = {
        'train_loss': self.train_loss.result(),
        'train_accuracy': self.train_accuracy.result(),
    }
    if self.use_performance and ((self.hvd is not None and self.hvd.rank() == 0) or self.hvd is None):
      run_time = time.time() - self.start
      self.time_list.append(run_time)
    if self.use_profiler and ((self.hvd is not None and self.hvd.rank() == 0) or self.hvd is None):
      profiler.stop()
    if (self.hvd is not None and self.hvd.rank() == 0) or self.hvd is None:
      self.time_callback.on_batch_end(self.epoch_helper.batch_index - 1)
      self._epoch_end()
    return metrics

  def train(self, num_steps: tf.Tensor):
    """See base class."""
    self.train_loop_begin()

    if self._train_loop_fn is None:
      self._train_loop_fn = _create_train_loop_fn(
          self.train_step, self.flags_obj.use_tf_while_loop,
          self.flags_obj.use_tf_function)

    if self._train_iter is None:
      self._train_iter = tf.nest.map_structure(iter, self._train_dataset)

    # Horovod: broadcast initial variable states from rank 0 to all other processes on batch 0.
    if self.flags_obj.use_horovod and (self.global_step.numpy() == 0):
      self._train_loop_fn(
              self._train_iter, tf.convert_to_tensor(1, dtype=tf.int32))
      num_steps = num_steps - 1
      self.hvd.broadcast_variables(self.model.trainable_variables, root_rank=0)
      self.hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)

    self._train_loop_fn(self._train_iter, num_steps)
    return self.train_loop_end()

  def eval_begin(self):
    """See base class."""
    self.test_loss.reset_states()
    self.test_accuracy.reset_states()
    self.test_accuracy_5.reset_states()

  def eval_step(self, iterator):
    """See base class."""

    def step_fn(inputs):
      """Function to run on the device."""
      images, labels = inputs
      logits = self.model(images, training=False)
      loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
      loss = tf.reduce_sum(loss) * (1.0 / self.total_batch_size)
      self.test_loss.update_state(loss)
      self.test_accuracy.update_state(labels, logits)
      self.test_accuracy_5.update_state(labels, logits)

    self.strategy.run(step_fn, args=(next(iterator),))

  def eval_end(self):
    """See base class."""
    return {
        'test_loss': self.test_loss.result(),
        'test_accuracy': self.test_accuracy.result(),
        'test_accuracy_5': self.test_accuracy_5.result()
    }

  def _epoch_begin(self):
    if self.epoch_helper.epoch_begin():
      self.time_callback.on_epoch_begin(self.epoch_helper.current_epoch)

  def _epoch_end(self):
    if self.epoch_helper.epoch_end():
      self.time_callback.on_epoch_end(self.epoch_helper.current_epoch)
