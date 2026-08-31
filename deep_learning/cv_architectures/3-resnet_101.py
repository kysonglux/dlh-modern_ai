#!/usr/bin/env python3
"""builds a ResNet-101 architecture as described in
Deep Residual Learning for Image Recognition (2015)"""
from tensorflow import keras


def build_resnet101(input_shape=(224, 224, 3), num_classes=1000):
    """builds a ResNet-101 architecture"""

    inputs = keras.Input(shape=input_shape)

    # Initial convolution and max pooling layers
    x = keras.layers.Conv2D(64, 7, strides=2, padding="same",
                            kernel_initializer="he_normal",
                            name="conv1")(inputs)
    x = keras.layers.BatchNormalization(axis=3, name="bn_conv1")(x)
    x = keras.layers.Activation("relu", name="conv1_relu")(x)
    x = keras.layers.MaxPooling2D(3, strides=2, padding="same",
                                  name="pool1")(x)

    # Stage 2
    x = make_layer(x, blocks=3, filters=64, stride=1, name="conv2")

    # Stage 3
    x = make_layer(x, blocks=4, filters=128, stride=2, name="conv3")

    # Stage 4
    x = make_layer(x, blocks=23, filters=256, stride=2, name="conv4")

    # Stage 5
    x = make_layer(x, blocks=3, filters=512, stride=2, name="conv5")

    x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax",
                                 kernel_initializer="he_normal",
                                 name="fc1000")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="ResNet101")


def bottleneck_block(x, filters, stride=1, downsample=False, name=None):
    shortcut = x

    # 1x1 reduce
    x = keras.layers.Conv2D(filters, 1, strides=stride,
                            kernel_initializer="he_normal",
                            name=f"{name}_conv1")(x)
    x = keras.layers.BatchNormalization(axis=3, name=f"{name}_bn1")(x)
    x = keras.layers.Activation("relu", name=f"{name}_relu1")(x)

    # 3x3 conv
    x = keras.layers.Conv2D(filters, 3, padding="same",
                            kernel_initializer="he_normal",
                            name=f"{name}_conv2")(x)
    x = keras.layers.BatchNormalization(axis=3, name=f"{name}_bn2")(x)
    x = keras.layers.Activation("relu", name=f"{name}_relu2")(x)

    # 1x1 expand
    x = keras.layers.Conv2D(filters * 4, 1,
                            kernel_initializer="he_normal",
                            name=f"{name}_conv3")(x)
    x = keras.layers.BatchNormalization(axis=3, name=f"{name}_bn3")(x)

    # 🔥 FIX: projection shortcut when downsampling or channel mismatch
    if downsample:
        shortcut = keras.layers.Conv2D(filters * 4, 1, strides=stride,
                                       kernel_initializer="he_normal",
                                       name=f"{name}_proj_conv")(shortcut)
        shortcut = keras.layers.BatchNormalization(
            axis=3, name=f"{name}_proj_bn")(shortcut)

    # Add
    x = keras.layers.Add(name=f"{name}_add")([x, shortcut])
    x = keras.layers.Activation("relu", name=f"{name}_out")(x)

    return x


def make_layer(x, blocks, filters, stride=1, name=None):
    x = bottleneck_block(x, filters, stride=stride, downsample=True,
                         name=f'{name}_block1')
    for i in range(1, blocks):
        x = bottleneck_block(x, filters, stride=1, downsample=False,
                             name=f'{name}_block{i+1}')
    return x
