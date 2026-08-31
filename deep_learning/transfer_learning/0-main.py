#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras
build_feature_extractor = __import__('0-frozen_extractor'
                                     ).build_feature_extractor
tf.random.set_seed(42)


def prepare_sample_batch(n=8):
    (x_train, _), _ = keras.datasets.cifar10.load_data()
    x = x_train[:n].astype("float32") / 255.0
    x = tf.image.resize(x, (224, 224))
    return x


extractor = build_feature_extractor()
print("Feature extractor summary:")
extractor.summary()

x = prepare_sample_batch(8)
features = extractor(x, training=False)
print("Input batch shape:", x.shape)
print("Extracted feature shape:", features.shape)
