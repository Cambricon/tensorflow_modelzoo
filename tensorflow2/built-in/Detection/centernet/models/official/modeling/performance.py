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

"""Functions and classes related to training performance."""

from absl import logging
import tensorflow as tf


def configure_optimizer(optimizer,
                        use_float16=False,
                        use_graph_rewrite=False,
                        loss_scale='dynamic',
                        use_experimental_api=False):
  """Configures optimizer object with performance options."""
  is_dynamic = False
  initial_scale = None
  if loss_scale == 'dynamic':
    is_dynamic = True
  else:
    initial_scale = loss_scale
  if use_float16:
    # TODO(b/171936854): Move all methods to non-experimental api.
      # Wraps optimizer with a LossScaleOptimizer. This is done automatically
      # in compile() with the "mixed_float16" policy, but since we do not call
      # compile(), we must wrap the optimizer manually.
    optimizer = (
        tf.keras.mixed_precision.LossScaleOptimizer(
            optimizer, dynamic=is_dynamic, initial_scale=initial_scale))

  if use_graph_rewrite:
    # Note: the model dtype must be 'float32', which will ensure
    # tf.keras.mixed_precision and enable_mixed_precision_graph_rewrite do not
    # double up.
    optimizer = (
        tf.compat.v1.mixed_precision.enable_mixed_precision_graph_rewrite(
            optimizer))
  return optimizer


def set_mixed_precision_policy(dtype, loss_scale=None,
                               use_experimental_api=False):
  """Sets mix precision policy."""
  if use_experimental_api:
    logging.warning('Passing use_experimental_api=True is deprecated. The '
                    'argument will be removed in the future.')
  assert use_experimental_api or loss_scale is None, (
      'loss_scale cannot be specified if use_experimental_api is False. If the '
      'non-experimental API is used, specify the loss scaling configuration '
      'when creating the LossScaleOptimizer instead.'
  )
  if dtype == tf.float16:
    # TODO(b/171936854): Move all methods to non-experimental api.
    policy = tf.keras.mixed_precision.Policy(
        'mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
  elif dtype == tf.bfloat16:
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
  elif dtype == tf.float32:
    tf.keras.mixed_precision.set_global_policy('float32')
  else:
    raise ValueError('Unexpected dtype: %s' % dtype)
