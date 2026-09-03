#!/usr/bin/env python3
"""data augmentation with keras"""
import tensorflow as tf


def build_data_augmentation():
    """builds a data augmentation layer"""
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=42),
            tf.keras.layers.RandomRotation(0.15, seed=42),
            tf.keras.layers.RandomZoom(0.15, seed=42),
            tf.keras.layers.RandomContrast(0.1, seed=42)
        ]
    )
    return data_augmentation
