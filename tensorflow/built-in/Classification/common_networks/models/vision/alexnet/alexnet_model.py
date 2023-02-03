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
"""AlexNet model for Keras.

Related papers/blogs:
- https://arxiv.org/abs/1512.03385
- https://arxiv.org/pdf/1603.05027v2.pdf
- http://torch.ch/blog/2016/02/04/resnets.html

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers
from models.vision.alexnet import imagenet_preprocessing

L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

layers = tf.keras.layers


def _gen_l2_regularizer(use_l2_regularizer=True):
  return regularizers.l2(L2_WEIGHT_DECAY) if use_l2_regularizer else None


def alexnet(num_classes,
            batch_size=None,
            use_l2_regularizer=True,
            rescale_inputs=False,
            dropout_keep_prob=0.5):
  """Instantiates the AlexNet architecture.
  
  Args:
    num_classes: `int` number of classes for image classification.
    batch_size: Size of the batches for each step.

    use_l2_regularizer: whether to use L2 regularizer on Conv/Dense layer.
    rescale_inputs: whether to rescale inputs from 0 to 1.
    dropout_keep_prob: the probility of dropout layer, default: 0.5

  Returns:
      A Keras model instance.
  """
  
  input_shape = (224, 224, 3)
  img_input = layers.Input(shape=input_shape, batch_size=batch_size)
  if rescale_inputs:
    # Hub image modules expect inputs in the range [0, 1]. This rescales these
    # inputs to the range expected by the trained model.
    x = layers.Lambda(
        lambda x: x * 255.0 - backend.constant(
            imagenet_preprocessing.CHANNEL_MEANS,
            shape=[1, 1, 3],
            dtype=x.dtype),
        name='rescale')(
            img_input)
  else:
    x = img_input

  if backend.image_data_format() == 'channels_first':
    x = layers.Permute((3, 1, 2))(x)
    bn_axis = 1
  else:  # channels_last
    bn_axis = 3
  x = layers.Conv2D(
      64, (11, 11),
      strides=(4, 4),
      padding='valid',
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='conv1')(
          x)
  x = layers.Activation('relu')(x)
  x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
  x = layers.Conv2D(
      192, (5, 5),
      strides=(1, 1),
      padding='same',
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='conv2')(
          x)
  x = layers.Activation('relu')(x)
  x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
  x = layers.Conv2D(
      384, (3, 3),
      strides=(1, 1),
      padding='same',
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='conv3')(
          x)
  x = layers.Activation('relu')(x)
  x = layers.Conv2D(
      384, (3, 3),
      strides=(1, 1),
      padding='same',
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='conv4')(
          x)
  x = layers.Activation('relu')(x)
  x = layers.Conv2D(
      256, (3, 3),
      strides=(1, 1),
      padding='same',
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='conv5')(
          x)
  x = layers.Activation('relu')(x)
  x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
  x = layers.Flatten()(x)
  x = layers.Dense(
       4096,
       kernel_initializer=initializers.RandomNormal(stddev=0.01),
       kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
       bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
       name='fc6')(
           x)
  x = layers.Activation('relu')(x)
  x = layers.Dropout(dropout_keep_prob)(x)
  x = layers.Dense(
       4096,
       kernel_initializer=initializers.RandomNormal(stddev=0.01),
       kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
       bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
       name='fc7')(
           x)
  x = layers.Activation('relu')(x)
  x = layers.Dropout(dropout_keep_prob)(x)
  x = layers.Dense(
       num_classes,
       kernel_initializer=initializers.RandomNormal(stddev=0.01),
       kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
       bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
       name='fc8')(
           x)
  x = layers.Activation('softmax', dtype='float32')(x)
   
  return models.Model(img_input, x, name='alexnet')
   

   


