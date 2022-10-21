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
"""Learning rate utilities for vision tasks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Mapping, Optional, List

import numpy as np
import tensorflow as tf

BASE_LEARNING_RATE = 0.1


class WarmupDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """A wrapper for LearningRateSchedule that includes warmup steps."""

  def __init__(self,
               lr_schedule: tf.keras.optimizers.schedules.LearningRateSchedule,
               warmup_steps: int,
               warmup_lr: Optional[float] = None,
               boundaries: List[int] = None,
               values: List[float] = None,
               name: str = None):
    """Add warmup decay to a learning rate schedule.

    Args:
      lr_schedule: base learning rate scheduler
      warmup_steps: number of warmup steps
      warmup_lr: an optional field for the final warmup learning rate. This
        should be provided if the base `lr_schedule` does not contain this
        field.
    """
    super(WarmupDecaySchedule, self).__init__()
    self._lr_schedule = lr_schedule
    self._warmup_steps = warmup_steps
    self._warmup_lr = warmup_lr
    self._boundaries = boundaries
    self._values = values
    self._name = name

  def __call__(self, step: int):
    lr = self._lr_schedule(step)
    if self._warmup_steps:
      if self._warmup_lr is not None:
        initial_learning_rate = tf.convert_to_tensor(
            self._warmup_lr, name="initial_learning_rate")
      else:
        initial_learning_rate = tf.convert_to_tensor(
            self._lr_schedule.initial_learning_rate,
            name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      global_step_recomp = tf.cast(step, dtype)
      warmup_steps = tf.cast(self._warmup_steps, dtype)
      warmup_lr = initial_learning_rate * global_step_recomp / warmup_steps
      lr = tf.cond(global_step_recomp < warmup_steps, lambda: warmup_lr,
                   lambda: lr)
    return lr

  def get_config(self) -> Mapping[str, Any]:
    config = self._lr_schedule.get_config()
    config.update({
        "lr_schedule": self._lr_schedule,
        "warmup_steps": self._warmup_steps,
        "warmup_lr": self._warmup_lr,
        "boundaries": self._boundaries,
        "values": self._values,
        "name": self._name,
    })
    return config


class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Class to generate learning rate tensor."""

  def __init__(self,
               batch_size: int,
               total_steps: int,
               warmup_steps: int,
               warmup_learning_rate: float = None,
               init_learning_rate: float = None):
    """Creates the consine learning rate tensor with linear warmup.

    Args:
      batch_size: The training batch size used in the experiment.
      total_steps: Total training steps.
      warmup_steps: Steps for the warm up period.
    """
    super(CosineDecayWithWarmup, self).__init__()
    base_lr_batch_size = 256
    self._batch_size = batch_size
    self._total_steps = total_steps
    self._init_learning_rate = BASE_LEARNING_RATE * batch_size / base_lr_batch_size
    self._warmup_steps = warmup_steps
    self._warmup_learning_rate = warmup_learning_rate

  def __call__(self, global_step: int):
    global_step = tf.cast(global_step, dtype=tf.float32)
    warmup_steps = self._warmup_steps
    init_lr = self._init_learning_rate
    total_steps = self._total_steps

    linear_warmup = global_step / warmup_steps * init_lr

    cosine_learning_rate = init_lr * (tf.cos(np.pi *
                                             (global_step - warmup_steps) /
                                             (total_steps - warmup_steps)) +
                                      1.0) / 2.0

    learning_rate = tf.where(global_step < warmup_steps, linear_warmup,
                             cosine_learning_rate)
    return learning_rate

  def get_config(self):
    return {
        "batch_size": self._batch_size,
        "total_steps": self._total_steps,
        "warmup_learning_rate": self._warmup_learning_rate,
        "warmup_steps": self._warmup_steps,
        "init_learning_rate": self._init_learning_rate,
    }

class PiecewiseConstantDecayWithWarmup(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  """Piecewise constant decay with warmup schedule."""

  def __init__(self,
               batch_size: int,
               epoch_size: int,
               warmup_epochs: int,
               boundaries: List[int],
               multipliers: List[float],
               rescaled_lr: float,
               step_boundaries: List[float],
               lr_values: List[float],
               warmup_steps: int):
    """Piecewise constant decay with warmup.

    Args:
      batch_size: The training batch size used in the experiment.
      epoch_size: The size of an epoch, or the number of examples in an epoch.
      warmup_epochs: The number of warmup epochs to apply.
      boundaries: The list of floats with strictly increasing entries.
      multipliers: The list of multipliers/learning rates to use for the
        piecewise portion. The length must be 1 less than that of boundaries.

    """
    super(PiecewiseConstantDecayWithWarmup, self).__init__()
    if len(boundaries) != len(multipliers) - 1:
      raise ValueError("The length of boundaries must be 1 less than the "
                       "length of multipliers")

    base_lr_batch_size = 256
    steps_per_epoch = epoch_size // batch_size
    self._rescaled_lr = BASE_LEARNING_RATE * batch_size / base_lr_batch_size
    self._step_boundaries = [float(steps_per_epoch) * x for x in boundaries]
    self._lr_values = [self._rescaled_lr * m for m in multipliers]
    self._warmup_steps = warmup_epochs * steps_per_epoch
    self._batch_size = batch_size
    self._epoch_size = epoch_size
    self._warmup_epochs = warmup_epochs
    self._boundaries = boundaries
    self._multipliers = multipliers

  def __call__(self, step: int):
    """Compute learning rate at given step."""
    def warmup_lr():
      return self._rescaled_lr * (
          tf.cast(step, tf.float32) / tf.cast(self._warmup_steps, tf.float32))
    def piecewise_lr():
      return tf.compat.v1.train.piecewise_constant(
          tf.cast(step, tf.float32), self._step_boundaries, self._lr_values)
    return tf.cond(step < self._warmup_steps, warmup_lr, piecewise_lr)

  def get_config(self) -> Mapping[str, Any]:
    return {
        "rescaled_lr": self._rescaled_lr,
        "step_boundaries": self._step_boundaries,
        "lr_values": self._lr_values,
        "warmup_steps": self._warmup_steps,
        "batch_size" : self._batch_size,
        "epoch_size" : self._epoch_size,
        "warmup_epochs" : self._warmup_epochs,
        "boundaries" : self._boundaries,
        "multipliers" : self._multipliers,
    }
