#!/usr/bin/env python3
"""builds a MobileNetV1 architecture as described in"""
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


def mobilenet_backbone(inputs):
    """builds a feature extractor using MobileNetV1 as backbone"""
    # Initial convolution layer
    X = keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same',
                            kernel_initializer='he_normal',
                            use_bias=False)(inputs)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.Activation('relu')(X)

    # Depthwise separable convolution blocks
    X = depthwise_separable_conv(X, 64, stride=1)
    X = depthwise_separable_conv(X, 128, stride=2)
    X = depthwise_separable_conv(X, 128, stride=1)
    X = depthwise_separable_conv(X, 256, stride=2)
    X = depthwise_separable_conv(X, 256, stride=1)
    X = depthwise_separable_conv(X, 512, stride=2)

    for _ in range(5):
        X = depthwise_separable_conv(X, 512, stride=1)

    X = depthwise_separable_conv(X, 1024, stride=2)
    X = depthwise_separable_conv(X, 1024, stride=1)

    return X


def mobilenet(input_shape=(224, 224, 3), num_classes=1000):
    """builds the MobileNetV1 backbone"""
    # Create the input layer
    inputs = keras.layers.Input(shape=input_shape)
    x = mobilenet_backbone(inputs)

    # Global average pooling
    x = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Final classifier
    outputs = keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions"
    )(x)

    model = keras.Model(inputs, outputs, name="MobileNetV1")
    return model
