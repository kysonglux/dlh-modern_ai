#!/usr/bin/env python3
"""loads a pretrained MobileNetV2 model and uses it as a feature extractor"""
from tensorflow import keras


def build_feature_extractor():
    """builds a feature extractor using MobileNetV2 as backbone"""

    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    feature_extractor = keras.Model(inputs, x)

    return feature_extractor
