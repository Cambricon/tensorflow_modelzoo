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
"""Inception V3 model for Keras.

Reference paper:
  - [Rethinking the Inception Architecture for Computer Vision](
      http://arxiv.org/abs/1512.00567) (CVPR 2016)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import layers
from tensorflow.python.keras.engine import training
from models.vision import imagenet_tools
#from tensorflow.python.keras.utils import data_utils
#from tensorflow.python.keras.utils import layer_utils
#from tensorflow.python.util.tf_export import keras_export

L2_WEIGHT_DECAY=1e-4
BATCH_NORM_DECAY=0.9
BATCH_NORM_EPSILON=1e-5

def _gen_l2_regularizer(use_l2_regularizer=True):
  return regularizers.l2(L2_WEIGHT_DECAY) if use_l2_regularizer else None
#@keras_export('keras.applications.inception_v3.InceptionV3',
#              'keras.applications.InceptionV3')


def InceptionV3(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    use_l2_regularizer=True,
    classes=1000,
    classifier_activation='softmax',
):
  """Instantiates the Inception v3 architecture.

  Reference paper:
  - [Rethinking the Inception Architecture for Computer Vision](
      http://arxiv.org/abs/1512.00567) (CVPR 2016)

  Optionally loads weights pre-trained on ImageNet.
  Note that the data format convention used by the model is
  the one specified in the `tf.keras.backend.image_data_format()`.

  Caution: Be sure to properly pre-process your inputs to the application.
  Please see `applications.inception_v3.preprocess_input` for an example.

  Arguments:
    include_top: Boolean, whether to include the fully-connected
      layer at the top, as the last layer of the network. Default to `True`.
    weights: One of `None` (random initialization),
      `imagenet` (pre-training on ImageNet),
      or the path to the weights file to be loaded. Default to `imagenet`.
    input_tensor: Optional Keras tensor (i.e. output of `layers.Input()`)
      to use as image input for the model. `input_tensor` is useful for sharing
      inputs between multiple different networks. Default to None.
    input_shape: Optional shape tuple, only to be specified
      if `include_top` is False (otherwise the input shape
      has to be `(299, 299, 3)` (with `channels_last` data format)
      or `(3, 299, 299)` (with `channels_first` data format).
      It should have exactly 3 inputs channels,
      and width and height should be no smaller than 75.
      E.g. `(150, 150, 3)` would be one valid value.
      `input_shape` will be ignored if the `input_tensor` is provided.
    pooling: Optional pooling mode for feature extraction
      when `include_top` is `False`.
      - `None` (default) means that the output of the model will be
          the 4D tensor output of the last convolutional block.
      - `avg` means that global average pooling
          will be applied to the output of the
          last convolutional block, and thus
          the output of the model will be a 2D tensor.
      - `max` means that global max pooling will be applied.
    classes: optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified. Default to 1000.
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
  #print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$', backend.image_data_format())
  if not (weights in {'imagenet', None} or os.path.exists(weights)):
    raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization), `imagenet` '
                     '(pre-training on ImageNet), '
                     'or the path to the weights file to be loaded.')

  if weights == 'imagenet' and include_top and classes != 1000:
    raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                     ' as true, `classes` should be 1000')

  # Determine proper input shape
  input_shape = (299,299,3)

  if input_tensor is None:
    img_input = layers.Input(shape=input_shape)
  else:
    if not backend.is_keras_tensor(input_tensor):
      img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor

  if backend.image_data_format() == 'channels_first':
    channel_axis = 1
  else:
    channel_axis = 3
    #print ('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%channel_last%%%%%%%%%%%%%%%%%%%%%%%%%')
    #img_input=layers.Permute((3,1,2))(img_input)

  x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
  x = conv2d_bn(x, 32, 3, 3, padding='valid')
  x = conv2d_bn(x, 64, 3, 3)
  x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

  x = conv2d_bn(x, 80, 1, 1, padding='valid')
  x = conv2d_bn(x, 192, 3, 3, padding='valid')
  x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

  # mixed 0: 35 x 35 x 256
  branch1x1 = conv2d_bn(x, 64, 1, 1)

  branch5x5 = conv2d_bn(x, 48, 1, 1)
  branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

  branch3x3dbl = conv2d_bn(x, 64, 1, 1)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

  branch_pool = layers.AveragePooling2D(
      (3, 3), strides=(1, 1), padding='same')(x)
  branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
  x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                         axis=channel_axis,
                         name='mixed0')

  # mixed 1: 35 x 35 x 288
  branch1x1 = conv2d_bn(x, 64, 1, 1)

  branch5x5 = conv2d_bn(x, 48, 1, 1)
  branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

  branch3x3dbl = conv2d_bn(x, 64, 1, 1)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

  branch_pool = layers.AveragePooling2D(
      (3, 3), strides=(1, 1), padding='same')(x)
  branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
  x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                         axis=channel_axis,
                         name='mixed1')

  # mixed 2: 35 x 35 x 288
  branch1x1 = conv2d_bn(x, 64, 1, 1)

  branch5x5 = conv2d_bn(x, 48, 1, 1)
  branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

  branch3x3dbl = conv2d_bn(x, 64, 1, 1)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

  branch_pool = layers.AveragePooling2D(
      (3, 3), strides=(1, 1), padding='same')(x)
  branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
  x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                         axis=channel_axis,
                         name='mixed2')

  # mixed 3: 17 x 17 x 768
  branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

  branch3x3dbl = conv2d_bn(x, 64, 1, 1)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
  branch3x3dbl = conv2d_bn(
      branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

  branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
  x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool],
                         axis=channel_axis,
                         name='mixed3')

  # mixed 4: 17 x 17 x 768
  branch1x1 = conv2d_bn(x, 192, 1, 1)

  branch7x7 = conv2d_bn(x, 128, 1, 1)
  branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
  branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

  branch7x7dbl = conv2d_bn(x, 128, 1, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

  branch_pool = layers.AveragePooling2D(
      (3, 3), strides=(1, 1), padding='same')(x)
  branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
  x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                         axis=channel_axis,
                         name='mixed4')

  # mixed 5, 6: 17 x 17 x 768
  for i in range(2):
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 160, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 160, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(
                                              x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                           axis=channel_axis,
                           name='mixed' + str(5 + i))

  # mixed 7: 17 x 17 x 768
  branch1x1 = conv2d_bn(x, 192, 1, 1)

  branch7x7 = conv2d_bn(x, 192, 1, 1)
  branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
  branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

  branch7x7dbl = conv2d_bn(x, 192, 1, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

  branch_pool = layers.AveragePooling2D(
      (3, 3), strides=(1, 1), padding='same')(x)
  branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
  x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                         axis=channel_axis,
                         name='mixed7')

  # mixed 8: 8 x 8 x 1280
  branch3x3 = conv2d_bn(x, 192, 1, 1)
  branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

  branch7x7x3 = conv2d_bn(x, 192, 1, 1)
  branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
  branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
  branch7x7x3 = conv2d_bn(
      branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

  branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
  x = layers.concatenate([branch3x3, branch7x7x3, branch_pool],
                         axis=channel_axis,
                         name='mixed8')

  # mixed 9: 8 x 8 x 2048
  for i in range(2):
    branch1x1 = conv2d_bn(x, 320, 1, 1)

    branch3x3 = conv2d_bn(x, 384, 1, 1)
    branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
    branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
    branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2],
                                   axis=channel_axis,
                                   name='mixed9_' + str(i))

    branch3x3dbl = conv2d_bn(x, 448, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
    branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
    branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
    branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2],
                                      axis=channel_axis)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(
                                              x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                           axis=channel_axis,
                           name='mixed' + str(9 + i))
  if include_top:
    # Classification block
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    imagenet_tools.validate_activation(classifier_activation, weights)
    x = layers.Dense(classes, activation=classifier_activation,
                     name='predictions')(x)
  else:
    if pooling == 'avg':
      x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
      x = layers.GlobalMaxPooling2D()(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  inputs = img_input
  # Create model.
  model = training.Model(inputs, x, name='inception_v3')

  # Load weights.
  if weights:
    model.load_weights(weights)

  return model


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None,
              use_l2_regularizer=True):
  """Utility function to apply conv + BN.

  Arguments:
    x: input tensor.
    filters: filters in `Conv2D`.
    num_row: height of the convolution kernel.
    num_col: width of the convolution kernel.
    padding: padding mode in `Conv2D`.
    strides: strides in `Conv2D`.
    name: name of the ops; will become `name + '_conv'`
      for the convolution and `name + '_bn'` for the
      batch norm layer.

  Returns:
    Output tensor after applying `Conv2D` and `BatchNormalization`.
  """
  if name is not None:
    bn_name = name + '_bn'
    conv_name = name + '_conv'
  else:
    bn_name = None
    conv_name = None
  if backend.image_data_format() == 'channels_first':
    bn_axis = 1
  else:
    bn_axis = 3

  x = layers.Conv2D(
      filters, (num_row, num_col),
      strides=strides,
      padding=padding,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      use_bias=False,
      name=conv_name)(
          x)
  x = layers.BatchNormalization(axis=bn_axis, 
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      scale=False, name=bn_name)(x)
  x = layers.Activation('relu', name=name)(x)
  return x



def inceptionv3(num_classes=1000, batch_size=None,
                use_l2_regularizer=True, rescale_inputs=False):
  input_tensor=layers.Input(shape=(299,299,3))
  return InceptionV3(input_tensor=input_tensor, weights=None)
#model=inceptionv3()
#print (model.summary())
'''
@keras_export('keras.applications.inception_v3.preprocess_input')
def preprocess_input(x, data_format=None):
  """Preprocesses a numpy array encoding a batch of images.

  Arguments
    x: A 4D numpy array consists of RGB values within [0, 255].

  Returns
    Preprocessed array.

  Raises
    ValueError: In case of unknown `data_format` argument.
  """
  return imagenet_utils.preprocess_input(x, data_format=data_format, mode='tf')


@keras_export('keras.applications.inception_v3.decode_predictions')
def decode_predictions(preds, top=5):
  """Decodes the prediction result from the model.

  Arguments
    preds: Numpy tensor encoding a batch of predictions.
    top: Integer, how many top-guesses to return.

  Returns
    A list of lists of top class prediction tuples
    `(class_name, class_description, score)`.
    One list of tuples per sample in batch input.

  Raises
    ValueError: In case of invalid shape of the `preds` array (must be 2D).
  """
  return imagenet_utils.decode_predictions(preds, top=top)


preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode='', ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_TF)
decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__'''
