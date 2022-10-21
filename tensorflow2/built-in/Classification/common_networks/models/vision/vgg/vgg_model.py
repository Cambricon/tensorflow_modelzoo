# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=invalid-name
"""VGG16 model for Keras."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import layers
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from models.vision import imagenet_tools


L2_WEIGHT_DECAY = 1e-4

def _gen_l2_regularizer(use_l2_regularizer=True):
  return regularizers.l2(L2_WEIGHT_DECAY) if use_l2_regularizer else None


def VGG16(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    use_l2_regularizer=True,
    classifier_activation='softmax',
):
  """Instantiates the VGG16 model.

  The default input size for this model is 224x224.

  Caution: Be sure to properly pre-process your inputs to the application.
  Please see `applications.vgg16.preprocess_input` for an example.

  Arguments:
      include_top: whether to include the 3 fully-connected
          layers at the top of the network.
      weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor
          (i.e. output of `layers.Input()`)
          to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
          if `include_top` is False (otherwise the input shape
          has to be `(224, 224, 3)`
          (with `channels_last` data format)
          or `(3, 224, 224)` (with `channels_first` data format).
          It should have exactly 3 input channels,
          and width and height should be no smaller than 32.
          E.g. `(200, 200, 3)` would be one valid value.
      pooling: Optional pooling mode for feature extraction
          when `include_top` is `False`.
          - `None` means that the output of the model will be
              the 4D tensor output of the
              last convolutional block.
          - `avg` means that global average pooling
              will be applied to the output of the
              last convolutional block, and thus
              the output of the model will be a 2D tensor.
          - `max` means that global max pooling will
              be applied.
      classes: optional number of classes to classify images
          into, only to be specified if `include_top` is True, and
          if no `weights` argument is specified.
      classifier_activation: A `str` or callable. The activation function to use
          on the "top" layer. Ignored unless `include_top=True`. Set
          `classifier_activation=None` to return the logits of the "top" layer.

  Returns:
    A `keras.Model` instance.

  Raises:
    ValueError: in case of invalid argument for `weights`,
      or invalid input shape.
    ValueError: if `classifier_activation` is not `softmax` or `None` when
      using a pretrained top layer.
  """
  # Determine proper input shape
  input_shape = imagenet_tools.obtain_input_shape(
      input_shape,
      default_size=224,
      min_size=32,
      data_format=backend.image_data_format(),
      require_flatten=include_top,
      weights=weights)
  if input_tensor is None:
    img_input = layers.Input(shape=input_shape)
  else:
    if not backend.is_keras_tensor(input_tensor):
      img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor
  # Block 1
  x=img_input
  if backend.image_data_format() == 'channels_first':
    x = layers.Permute((3, 1, 2))(x)
    bn_axis = 1
  else:  # channels_last
    bn_axis = 3
  x = layers.Conv2D(
      64, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block1_conv1')(x)
  x = layers.Conv2D(
      64, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block1_conv2')(x)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

  # Block 2
  x = layers.Conv2D(
      128, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block2_conv1')(x)
  x = layers.Conv2D(
      128, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block2_conv2')(x)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

  # Block 3
  x = layers.Conv2D(
      256, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block3_conv1')(x)
  x = layers.Conv2D(
      256, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block3_conv2')(x)
  x = layers.Conv2D(
      256, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block3_conv3')(x)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

  # Block 4
  x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block4_conv1')(x)
  x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block4_conv2')(x)
  x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block4_conv3')(x)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), 
      name='block4_pool')(x)

  # Block 5
  x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block5_conv1')(x)
  x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block5_conv2')(x)
  x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block5_conv3')(x)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

  if include_top:
    # Classification block
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(4096, activation='relu',
      kernel_initializer=initializers.RandomNormal(stddev=0.01),
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='fc1')(x)
    x = layers.Dense(4096, activation='relu', 
      kernel_initializer=initializers.RandomNormal(stddev=0.01),
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='fc2')(x)

    imagenet_tools.validate_activation(classifier_activation, weights)
    x = layers.Dense(classes, activation=classifier_activation,
      kernel_initializer=initializers.RandomNormal(stddev=0.01),
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                     name='predictions', dtype='float32')(x)
  else:
    if pooling == 'avg':
      x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
      x = layers.GlobalMaxPooling2D()(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = layer_utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input
  # Create model.
  model = training.Model(inputs, x, name='vgg16')

  return model


def VGG19(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    use_l2_regularizer=True,
    classifier_activation='softmax',
):
  """Instantiates the VGG19 model.

  The default input size for this model is 224x224.

  Caution: Be sure to properly pre-process your inputs to the application.
  Please see `applications.vgg19.preprocess_input` for an example.

  Arguments:
      include_top: whether to include the 3 fully-connected
          layers at the top of the network.
      weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor
          (i.e. output of `layers.Input()`)
          to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
          if `include_top` is False (otherwise the input shape
          has to be `(224, 224, 3)`
          (with `channels_last` data format)
          or `(3, 224, 224)` (with `channels_first` data format).
          It should have exactly 3 input channels,
          and width and height should be no smaller than 32.
          E.g. `(200, 200, 3)` would be one valid value.
      pooling: Optional pooling mode for feature extraction
          when `include_top` is `False`.
          - `None` means that the output of the model will be
              the 4D tensor output of the
              last convolutional block.
          - `avg` means that global average pooling
              will be applied to the output of the
              last convolutional block, and thus
              the output of the model will be a 2D tensor.
          - `max` means that global max pooling will
              be applied.
      classes: optional number of classes to classify images
          into, only to be specified if `include_top` is True, and
          if no `weights` argument is specified.
      classifier_activation: A `str` or callable. The activation function to use
          on the "top" layer. Ignored unless `include_top=True`. Set
          `classifier_activation=None` to return the logits of the "top" layer.

  Returns:
    A `keras.Model` instance.

  Raises:
    ValueError: in case of invalid argument for `weights`,
      or invalid input shape.
    ValueError: if `classifier_activation` is not `softmax` or `None` when
      using a pretrained top layer.
  """
  # Determine proper input shape
  input_shape = imagenet_tools.obtain_input_shape(
      input_shape,
      default_size=224,
      min_size=32,
      data_format=backend.image_data_format(),
      require_flatten=include_top,
      weights=weights)
  if input_tensor is None:
    img_input = layers.Input(shape=input_shape)
  else:
    if not backend.is_keras_tensor(input_tensor):
      img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor
  # Block 1
  x=img_input
  if backend.image_data_format() == 'channels_first':
    x = layers.Permute((3, 1, 2))(x)
    bn_axis = 1
  else:  # channels_last
    bn_axis = 3
  x = layers.Conv2D(
      64, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block1_conv1')(x)
  x = layers.Conv2D(
      64, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block1_conv2')(x)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

  # Block 2
  x = layers.Conv2D(
      128, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block2_conv1')(x)
  x = layers.Conv2D(
      128, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block2_conv2')(x)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

  # Block 3
  x = layers.Conv2D(
      256, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block3_conv1')(x)
  x = layers.Conv2D(
      256, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block3_conv2')(x)
  x = layers.Conv2D(
      256, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block3_conv3')(x)
  x = layers.Conv2D(
      256, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block3_conv4')(x)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

  # Block 4
  x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block4_conv1')(x)
  x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block4_conv2')(x)
  x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block4_conv3')(x)
  x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block4_conv4')(x)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), 
      name='block4_pool')(x)

  # Block 5
  x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block5_conv1')(x)
  x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block5_conv2')(x)
  x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block5_conv3')(x)
  x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', 
      use_bias=True,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block5_conv4')(x)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

  if include_top:
    # Classification block
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(4096, activation='relu',
      kernel_initializer=initializers.RandomNormal(stddev=0.01),
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='fc1')(x)
    x = layers.Dense(4096, activation='relu', 
      kernel_initializer=initializers.RandomNormal(stddev=0.01),
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='fc2')(x)

    imagenet_tools.validate_activation(classifier_activation, weights)
    x = layers.Dense(classes, activation=classifier_activation,
      kernel_initializer=initializers.RandomNormal(stddev=0.01),
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                     name='predictions', dtype='float32')(x)
  else:
    if pooling == 'avg':
      x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
      x = layers.GlobalMaxPooling2D()(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = layer_utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input
  # Create model.
  model = training.Model(inputs, x, name='vgg19')

  return model

def vgg16(num_classes=1000, batch_size=None, use_l2_regularizer=True, rescale_inputs=False):
    input_tensor=layers.Input(shape=(224,224,3), batch_size=batch_size)
    return VGG16(input_tensor=input_tensor)

def vgg19(num_classes=1000, batch_size=None, use_l2_regularizer=True, rescale_inputs=False):
    input_tensor=layers.Input(shape=(224,224,3), batch_size=batch_size)
    return VGG19(input_tensor=input_tensor)
