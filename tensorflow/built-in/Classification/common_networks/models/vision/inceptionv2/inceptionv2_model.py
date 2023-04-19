from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.keras import regularizers
from tensorflow.python.keras.initializers import TruncatedNormal
from tensorflow.python.keras import backend
from tensorflow.python.keras.layers import Conv2D, Dense, Input, concatenate, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, AveragePooling2D, \
    Activation, BatchNormalization
from tensorflow.python.keras import models
from tensorflow.python.keras.utils import layer_utils
from models.vision import imagenet_tools

trunc_normal = lambda stddev: TruncatedNormal(0.0, stddev)

L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              strides=(1, 1),
              padding='same',
              data_format='channels_last',
              kernel_initializer=None,
              use_l2_regularizer=True,
              weight_decay=L2_WEIGHT_DECAY,
              momentum=BATCH_NORM_DECAY,
              epsilon=BATCH_NORM_EPSILON,
              name=None):
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
        conv_name = name + '_conv'
        bn_name = name + '_bn'
        act_name = name
    else:
        bn_name = None
        act_name = None
    if data_format == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    if weight_decay:
        kernel_regularizer = regularizers.l2(weight_decay)
    else:
        kernel_regularizer = None

    if kernel_initializer is None:
        kernel_initializer = "he_normal"

    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        name=conv_name)(x)
    x = BatchNormalization(momentum=momentum, epsilon=epsilon,
                           axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=act_name)(x)
    return x

def inceptionv2(include_top=True,
                batch_size=None,
                use_l2_regularizer=True,
                weights=None,
                rescale_inputs=False,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                num_classes=1000,
                classifier_activation='softmax',
                concat_axis=3,
                data_format='channels_last'):

    if weights is not None and not os.path.exists(weights):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')
    print("num_classes", num_classes)
    if include_top and num_classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    input_shape = imagenet_tools.obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=75,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape, batch_size=batch_size)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if rescale_inputs:
        # Hub image modules expect inputs in the range [0, 1]. This rescales these
        # inputs to the range expected by the trained model.
        img_input = layers.Lambda(
            lambda x: x * 255.0 - backend.constant(
                imagenet_preprocessing.CHANNEL_MEANS,
                shape=[1, 1, 3],
                dtype=x.dtype),
            name='rescale')(img_input)

    x = conv2d_bn(img_input, filters=64, num_row=7, num_col=7, strides=(2, 2),
                  use_l2_regularizer=use_l2_regularizer,
                  kernel_initializer=trunc_normal(1.0), name='Conv2d_1a_7x7')  # 112 112 64
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same',
                     data_format=data_format, name='MaxPool_2a_3x3')(x)  # 56 56 64
    x = conv2d_bn(x, filters=64, num_row=1, num_col=1,
                  use_l2_regularizer=use_l2_regularizer,
                  kernel_initializer=trunc_normal(1.0), name='Conv2d_2b_1x1')  # 56 56 64
    x = conv2d_bn(x, filters=192, num_row=3, num_col=3,
                  use_l2_regularizer=use_l2_regularizer,
                  name='Conv2d_2c_3x3')  # 56 56 192

    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same',
                     data_format=data_format, name='MaxPool_3a_3x3')(x)  # 28 28 192

    # Inception Module
    # Mixed_3b
    prefix = "Mixed_3b_"
    # Branch_0
    branch_0 = conv2d_bn(x, filters=64, num_row=1, num_col=1,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_0_Conv2d_0a_1x1")  # 28 28 64

    # Branch_1
    branch_1 = conv2d_bn(x, filters=64, num_row=1, num_col=1,
                         kernel_initializer=trunc_normal(0.09),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_1_Conv2d_0a_1x1")  # 28 28 64
    branch_1 = conv2d_bn(branch_1, filters=64, num_row=3, num_col=3,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_1_Conv2d_0b_3x3")  # 28 28 64

    # Branch_2
    branch_2 = conv2d_bn(x, filters=64, num_row=1, num_col=1,
                         kernel_initializer=trunc_normal(0.09),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_2_Conv2d_0a_1x1")  # 28 28 64
    branch_2 = conv2d_bn(branch_2, filters=96, num_row=3, num_col=3,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_2_Conv2d_0b_3x3")  # 28 28 96
    branch_2 = conv2d_bn(branch_2, filters=96, num_row=3, num_col=3,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_2_Conv2d_0c_3x3")  # 28 28 96

    # Branch_3
    branch_3 = AveragePooling2D(pool_size=(3, 3),
                                strides=1, padding='same', data_format=data_format,
                                name=prefix + 'Branch_3_AvgPool_0a_3x3')(x)  # 28 28 192
    branch_3 = conv2d_bn(branch_3, filters=32, num_row=1, num_col=1,
                         kernel_initializer=trunc_normal(0.1),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_3_Conv2d_0b_1x1")  # 28 28 32

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=concat_axis, name=prefix + "_Concat")  # 28 28 256

    # Mixed_3c
    prefix = "Mixed_3c_"
    # Branch_0
    branch_0 = conv2d_bn(x, filters=64, num_row=1, num_col=1,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_0_Conv2d_0a_1x1")  # 28 28 64

    # Branch_1
    branch_1 = conv2d_bn(x, filters=64, num_row=1, num_col=1,
                         kernel_initializer=trunc_normal(0.09),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_1_Conv2d_0a_1x1")  # 28 28 64

    branch_1 = conv2d_bn(branch_1, filters=96, num_row=3, num_col=3,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_1_Conv2d_0b_3x3")  # 28 28 96

    # Branch_2
    branch_2 = conv2d_bn(x, filters=64, num_row=1, num_col=1,
                         kernel_initializer=trunc_normal(0.09),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_2_Conv2d_0a_1x1")  # 28 28 64
    branch_2 = conv2d_bn(branch_2, filters=96, num_row=3, num_col=3,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_2_Conv2d_0b_3x3")  # 28 28 96
    branch_2 = conv2d_bn(branch_2, filters=96, num_row=3, num_col=3,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_2_Conv2d_0c_3x3")  # 28 28 96

    # Branch_3
    branch_3 = AveragePooling2D(pool_size=(3, 3),
                                strides=1, padding='same', data_format=data_format,
                                name=prefix + 'Branch_3_AvgPool_0a_3x3')(x)  # 28 28 256
    branch_3 = conv2d_bn(branch_3, filters=64, num_row=1, num_col=1,
                         kernel_initializer=trunc_normal(0.1),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_3_Conv2d_0b_1x1")  # 28 28 64

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=concat_axis, name=prefix + "_Concat")  # 28 28 320

    # Mixed_4a
    prefix = "Mixed_4a_"
    branch_0 = conv2d_bn(x, filters=128, num_row=1, num_col=1,
                         kernel_initializer=trunc_normal(0.09),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_0_Conv2d_0a_1x1")  # 28 28 128
    branch_0 = conv2d_bn(branch_0, filters=160, num_row=3, num_col=3, strides=(2, 2),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_0_Conv2d_1a_3x3")  # 14 14 160

    # Branch_1
    branch_1 = conv2d_bn(x, filters=64, num_row=1, num_col=1,
                         kernel_initializer=trunc_normal(0.09),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_1_Conv2d_0a_1x1")  # 28 28 64
    branch_1 = conv2d_bn(branch_1, filters=96, num_row=3, num_col=3, strides=(2, 2),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_1_Conv2d_0b_1x1")  # 14 14 96

    # Branch_2
    branch_2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same',
                     data_format=data_format, name=prefix + 'Branch_2_MaxPool_1a_3x3')(x)  # 14 14 320

    x = concatenate([branch_0, branch_1, branch_2], axis=concat_axis)  # 14 14 576

    # Mixed_4b
    prefix = "Mixed_4b_"
    # Branch_0
    branch_0 = conv2d_bn(x, filters=224, num_col=1, num_row=1,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_0_Conv2d_0a_1x1")  # 14 14 224

    # Branch_1
    branch_1 = conv2d_bn(x, filters=64, num_col=1, num_row=1,
                         kernel_initializer=trunc_normal(0.09),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_1_Conv2d_0a_1x1")  # 14 14 64
    branch_1 = conv2d_bn(branch_1, filters=96, num_col=3, num_row=3,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_1_Conv2d_0b_3x3")  # 14 14 96

    # Branch_2
    branch_2 = conv2d_bn(x, filters=96, num_col=1, num_row=1,
                         kernel_initializer=trunc_normal(0.09),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_2_Conv2d_0a_1x1")  # 14 14 96
    branch_2 = conv2d_bn(branch_2, filters=128, num_col=3, num_row=3,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_2_Conv2d_0b_3x3")  # 14 14 128
    branch_2 = conv2d_bn(branch_2, filters=128, num_col=3, num_row=3,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_2_Conv2d_0c_3x3")  # 14 14 128

    # Branch_3
    branch_3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name=prefix + "Branch_3_AvgPool_0a_3x3")(x)  # 14 14 576
    branch_3 = conv2d_bn(branch_3, filters=128, num_col=1, num_row=1,
                         kernel_initializer=trunc_normal(0.1),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_3_Conv2d_0b_1x1")  # 14 14 128

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=concat_axis, name=prefix + '_Concat')  # 14 14 576

    # Mixed_4c
    prefix = "Mixed_4c_"
    # Branch_0
    branch_0 = conv2d_bn(x, filters=192, num_col=1, num_row=1,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_0_Conv2d_0a_1x1")  # 14 14 192

    # Branch_1
    branch_1 = conv2d_bn(x, filters=96, num_col=1, num_row=1,
                         kernel_initializer=trunc_normal(0.09),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_1_Conv2d_0a_1x1")  # 14 14 96
    branch_1 = conv2d_bn(branch_1, filters=128, num_col=3, num_row=3,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_1_Conv2d_0b_3x3")  # 14 14 128

    # Branch_2
    branch_2 = conv2d_bn(x, filters=96, num_col=1, num_row=1,
                         kernel_initializer=trunc_normal(0.09),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_2_Conv2d_0a_1x1")  # 14 14 96
    branch_2 = conv2d_bn(branch_2, filters=128, num_col=3, num_row=3,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_2_Conv2d_0b_3x3")  # 14 14 128
    branch_2 = conv2d_bn(branch_2, filters=128, num_col=3, num_row=3,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_2_Conv2d_0c_3x3")  # 14 14 128

    # Branch_3
    branch_3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same',
                                name=prefix + "Branch_3_AvgPool_0a_3x3")(x)  # 14 14 576
    branch_3 = conv2d_bn(branch_3, filters=128, num_col=1, num_row=1,
                         kernel_initializer=trunc_normal(0.1),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_3_Conv2d_0b_1x1")  # 14 14 128
    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=concat_axis, name=prefix + '_Concat')  # 14 14 576

    # Mixed_4c
    prefix = "Mixed_4d_"
    # Branch_0
    branch_0 = conv2d_bn(x, filters=160, num_col=1, num_row=1,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_0_Conv2d_0a_1x1")  # 14 14 160

    # Branch_1
    branch_1 = conv2d_bn(x, filters=128, num_col=1, num_row=1,
                         kernel_initializer=trunc_normal(0.09),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_1_Conv2d_0a_1x1")  # 14 14 128
    branch_1 = conv2d_bn(branch_1, filters=160, num_col=3, num_row=3,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_1_Conv2d_0b_3x3")  # 14 14 160

    # Branch_2
    branch_2 = conv2d_bn(x, filters=128, num_col=1, num_row=1,
                         kernel_initializer=trunc_normal(0.09),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_2_Conv2d_0a_1x1")  # 14 14 128
    branch_2 = conv2d_bn(branch_2, filters=160, num_col=3, num_row=3,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_2_Conv2d_0b_3x3")  # 14 14 160
    branch_2 = conv2d_bn(branch_2, filters=160, num_col=3, num_row=3,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_2_Conv2d_0c_3x3")  # 14 14 160

    # Branch_3
    branch_3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same',
                                name=prefix + "Branch_3_AvgPool_0a_3x3")(x)  # 14 14 576
    branch_3 = conv2d_bn(branch_3, filters=96, num_col=1, num_row=1,
                         kernel_initializer=trunc_normal(0.1),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_3_Conv2d_0b_1x1")  # 14 14 96

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=concat_axis, name=prefix + "_Concat")  # 14 14 576

    # Mixed_4e
    prefix = "Mixed_4e_"
    # Branch_0
    branch_0 = conv2d_bn(x, filters=96, num_col=1, num_row=1,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_0_Conv2d_0a_1x1")  # 14 14 96

    # Branch_1
    branch_1 = conv2d_bn(x, filters=128, num_col=1, num_row=1,
                         kernel_initializer=trunc_normal(0.09),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_1_Conv2d_0a_1x1")  # 14 14 128
    branch_1 = conv2d_bn(branch_1, filters=192, num_col=3, num_row=3,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_1_Conv2d_0b_3x3")  # 14 14 192

    # Branch_2
    branch_2 = conv2d_bn(x, filters=160, num_col=1, num_row=1,
                         kernel_initializer=trunc_normal(0.09),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_2_Conv2d_0a_1x1")  # 14 14 160
    branch_2 = conv2d_bn(branch_2, filters=192, num_col=3, num_row=3,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_2_Conv2d_0b_3x3")  # 14 14 192
    branch_2 = conv2d_bn(branch_2, filters=192, num_col=3, num_row=3,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_2_Conv2d_0c_3x3")  # 14 14 192

    # Branch_3
    branch_3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same',
                                name=prefix + "Branch_3_AvgPool_0a_3x3")(x)  # 14 14 576
    branch_3 = conv2d_bn(branch_3, filters=96, num_col=1, num_row=1,
                         kernel_initializer=trunc_normal(0.1),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_3_Conv2d_0b_1x1")  # 14 14 96

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=concat_axis, name=prefix + "_Concat")  # 14 14 576

    # Mixed_5a
    prefix = "Mixed_5a_"
    # Branch_0
    branch_0 = conv2d_bn(x, filters=128, num_col=1, num_row=1,
                         kernel_initializer=trunc_normal(0.09),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_0_Conv2d_0a_1x1")  # 14 14 128
    branch_0 = conv2d_bn(branch_0, filters=192, num_col=3, num_row=3, strides=(2, 2),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_0_Conv2d_1a_3x3")  # 7 7 192

    # Branch_1
    branch_1 = conv2d_bn(x, filters=192, num_col=1, num_row=1,
                         kernel_initializer=trunc_normal(0.09),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_1_Conv2d_0a_1x1")  # 14 14 192
    branch_1 = conv2d_bn(branch_1, filters=256, num_col=3, num_row=3,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_1_Conv2d_0b_3x3")  # 14 14 256
    branch_1 = conv2d_bn(branch_1, filters=256, num_col=3, num_row=3, strides=(2, 2),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_1_Conv2d_1a_3x3")  # 7 7 256

    # Branch_2
    branch_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name=prefix + "Branch_2_MaxPool_1a_3x3")(x)  # 7 7 576

    x = concatenate([branch_0, branch_1, branch_2], concat_axis, name=prefix + "_Concat")  # 7 7 1024

    # Mixed_5b
    prefix = "Mixed_5b_"
    # Branch_0
    branch_0 = conv2d_bn(x, filters=352, num_col=1, num_row=1,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_0_Conv2d_0a_1x1")  # 7 7 352

    # Branch_1
    branch_1 = conv2d_bn(x, filters=192, num_col=1, num_row=1,
                         kernel_initializer=trunc_normal(0.09),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_1_Conv2d_0a_1x1")  # 7 7 192
    branch_1 = conv2d_bn(branch_1, filters=320, num_col=3, num_row=3,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_1_Conv2d_0b_3x3")  # 7 7 320

    # Branch_2
    branch_2 = conv2d_bn(x, filters=160, num_col=1, num_row=1,
                         kernel_initializer=trunc_normal(0.09),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_2_Conv2d_0a_1x1")  # 7 7 160
    branch_2 = conv2d_bn(branch_2, filters=224, num_col=3, num_row=3,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_2_Conv2d_0b_3x3")  # 7 7 224
    branch_2 = conv2d_bn(branch_2, filters=224, num_col=3, num_row=3,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_2_Conv2d_0c_3x3")  # 7 7 224

    # Branch_3
    branch_3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name=prefix + "Branch_3_AvgPool_0a_3x3")(x)  # 7 7 1024
    branch_3 = conv2d_bn(branch_3, filters=128, num_col=1, num_row=1,
                         kernel_initializer=trunc_normal(0.1),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_3_Conv2d_0b_1x1")  # 7 7 128

    x = concatenate([branch_0, branch_1, branch_2, branch_3], concat_axis, name=prefix + "_Concat")  # 7 7 1024

    # Mixed_5c
    prefix = "Mixed_5c_"
    # Branch_0
    branch_0 = conv2d_bn(x, filters=352, num_col=1, num_row=1,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_0_Conv2d_0a_1x1")  # 7 7 352

    # Branch_1
    branch_1 = conv2d_bn(x, filters=192, num_col=1, num_row=1,
                         kernel_initializer=trunc_normal(0.09),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_1_Conv2d_0a_1x1")  # 7 7 192
    branch_1 = conv2d_bn(branch_1, filters=320, num_col=3, num_row=3,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_1_Conv2d_0b_3x3")  # 7 7 320

    # Branch_2
    branch_2 = conv2d_bn(x, filters=192, num_col=1, num_row=1,
                         kernel_initializer=trunc_normal(0.09),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_2_Conv2d_0a_1x1")  # 7 7 192
    branch_2 = conv2d_bn(branch_2, filters=224, num_col=3, num_row=3,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_2_Conv2d_0b_3x3")  # 7 7 224
    branch_2 = conv2d_bn(branch_2, filters=224, num_col=3, num_row=3,
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_2_Conv2d_0c_3x3")  # 7 7 224

    # Branch_3
    branch_3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name=prefix + "Branch_3_AvgPool_0a_3x3")(x)  # 7 7 1024
    branch_3 = conv2d_bn(branch_3, filters=128, num_col=1, num_row=1,
                         kernel_initializer=trunc_normal(0.1),
                         use_l2_regularizer=use_l2_regularizer,
                         name=prefix + "Branch_3_Conv2d_0b_1x1")  # 7 7 128

    x = concatenate([branch_0, branch_1, branch_2, branch_3], concat_axis, name=prefix + "_Concat")  # 7 7 1024

    # Head
    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        imagenet_tools.validate_activation(classifier_activation, weights)
        x = Dense(num_classes, activation=classifier_activation, name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account any potential preecessors of 'input_tensor'.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    return models.Model(inputs, x, name='inceptionv2')
