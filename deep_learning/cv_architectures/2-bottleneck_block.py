#!/usr/bin/env python3
"""implements a ResNet bottleneck block"""
from tensorflow import keras


def bottleneck_block(x, filters, stride=1, downsample=False,
                     name=None):
    """builds a bottleneck block as described in
    Deep Residual Learning for Image Recognition (2015)

    Args:
        X: input tensor of shape (m, h, w, c) where m is the number of
            examples, h is the height in pixels, w is the width in pixels,
            and c is the number of channels
        filters: integer, number of filters in the first 1x1 convolution
        stride: integer, stride to be used
        downsample: boolean, whether to downsample in the second convolution
        name: string, name of the block

    Returns:
        The activated output of the bottleneck block
    """

    F1 = 16
    F2 = 16
    F3 = 64

    shortcut = x

    out = keras.layers.Conv2D(
        F1, 1, strides=1, padding="valid",
        kernel_initializer="he_normal",
        name=f"{name}_conv1"
    )(x)
    out = keras.layers.BatchNormalization(axis=3, name=f"{name}_bn1")(out)
    out = keras.layers.Activation("relu", name=f"{name}_relu1")(out)

    # Conv2 (3×3)
    out = keras.layers.Conv2D(
        F2, 3, strides=1, padding="same",
        kernel_initializer="he_normal",
        name=f"{name}_conv2"
    )(out)
    out = keras.layers.BatchNormalization(axis=3, name=f"{name}_bn2")(out)
    out = keras.layers.Activation("relu", name=f"{name}_relu2")(out)

    # Conv3 (expand)
    out = keras.layers.Conv2D(
        F3, 1, strides=1, padding="valid",
        kernel_initializer="he_normal",
        name=f"{name}_conv3"
    )(out)
    out = keras.layers.BatchNormalization(axis=3, name=f"{name}_bn3")(out)

    # Shortcut (identity)
    out = keras.layers.Add(name=f"{name}_add")([out, shortcut])
    out = keras.layers.Activation("relu", name=f"{name}_out")(out)

    return out
