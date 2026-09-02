#!/usr/bin/env python3
"""builds a ResNet-101 architecture as described in
Deep Residual Learning for Image Recognition (2015)"""
from tensorflow import keras
bottleneck_block = __import__('2-bottleneck_block').bottleneck_block


def make_layer(x, blocks, filters, stride=1, name=None):
    """builds a layer of bottleneck blocks"""
    x = bottleneck_block(x, filters, stride=stride, downsample=True,
                         name=f'{name}_block1')
    for i in range(1, blocks):
        x = bottleneck_block(x, filters, stride=1, downsample=False,
                             name=f'{name}_block{i+1}')
    return x


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
