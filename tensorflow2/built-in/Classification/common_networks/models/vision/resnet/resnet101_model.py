from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import initializers
from tensorflow.keras import models
from tensorflow.keras import regularizers
from models.vision.resnet import imagenet_preprocessing

L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY= 0.9
BATCH_NORM_EPSILON = 1e-5

layers = tf.keras.layers

def _gen_l2_regularizer(use_l2_regularizer=True):
  return regularizers.l2(L2_WEIGHT_DECAY) if use_l2_regularizer else None

keras_utils = None


def block1(x, filters, kernel_size=3, stride=1,
           conv_shortcut=True, name=None, use_l2_regularizer=True):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut is True:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride,
                                 padding='same',use_bias=False, kernel_initializer='he_normal',
                                 kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                                 name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_DECAY,
                                             name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv',
                      padding='same',use_bias=False, kernel_initializer='he_normal',
                      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                     )(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_DECAY,
                                  name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='SAME',
                      use_bias=False, kernel_initializer='he_normal',
                      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                      name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_DECAY,
                                  name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1,
                      padding='same',use_bias=False, kernel_initializer='he_normal',
                      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                      name=name + '_3_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_DECAY,
                                  name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack1(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.

    # Returns
        Output tensor for the stacked blocks.
    """
    x = block1(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x


def ResNet(stack_fn,
           preact,
           use_bias,
           batch_size,
           model_name='resnet',
           include_top=True,
           input_tensor=None,
           input_shape=None,
           pooling=None,
           classes=1000,
           rescale_inputs=False,
           use_l2_regularizer=True,
           **kwargs):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        stack_fn: a function that returns output tensor for the
            stacked residual blocks.
        preact: whether to use pre-activation or not
            (True for ResNetV2, False for ResNet and ResNeXt).
        use_bias: whether to use biases for convolutional layers or not
            (True for ResNet and ResNetV2, False for ResNeXt).
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    input_shape = (224,224,3)
    img_input = layers.Input(shape=input_shape, batch_size=batch_size)
    if rescale_inputs:
      x = layers.Lambda(
          lambda x: x*255.0 - backend.constant(
            imagenet_preprocessing.CHANNEL_MEANS,
            shape=[1, 1, 3],
            dtype=x.dtype), name='rescale')(img_input)
    else:
      x = img_input
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    if backend.image_data_format() == "channels_first":
      x = layers.Permute((3,1,2))(x)
      bn_axis=1
    else:
      bn_axis=3
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(x)
    x = layers.Conv2D(64, 7, strides=2, name='conv1_conv',
                      use_bias=False, kernel_initializer='he_normal',
                      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                     )(x)

    if preact is False:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name='conv1_bn')(x)
        x = layers.Activation('relu', name='conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)

    if preact is True:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name='post_bn')(x)
        x = layers.Activation('relu', name='post_relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='probs',
                         kernel_initializer=initializers.RandomNormal(stddev=0.01),
                         kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                         bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                         dtype='float32')(x)

    # Create model.
    model = models.Model(img_input, x, name=model_name)

    return model


def resnet101(num_classes,
              batch_size=None,
              use_l2_regularizer=True,
              include_top=True,
              input_tensor=None,
              input_shape=(224,224,3),
              **kwargs):
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name='conv2')
        x = stack1(x, 128, 4, name='conv3')
        x = stack1(x, 256, 23, name='conv4')
        x = stack1(x, 512, 3, name='conv5')
        return x
    return ResNet(stack_fn, False, True, batch_size, 'resnet101',
                  include_top,
                  input_tensor, input_shape,
                  num_classes,
                  **kwargs)
