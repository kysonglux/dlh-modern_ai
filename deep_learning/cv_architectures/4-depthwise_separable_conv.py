#!/usr/bin/env python3
"""A depthwise separable convolution block, MobileNetV1"""
from tensorflow import keras


def depthwise_separable_conv(X, filters, stride=1):
    """builds a depthwise separable convolution block"""
    # Depthwise convolution
    X = keras.layers.DepthwiseConv2D(kernel_size=3, strides=stride,
                                     padding='same', depth_multiplier=1,
                                     kernel_initializer='he_normal',
                                     use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.Activation('relu')(X)

    # Pointwise convolution
    X = keras.layers.Conv2D(filters, kernel_size=1, strides=1,
                            padding='same', kernel_initializer='he_normal',
                            use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.Activation('relu')(X)

    return X
