#!/usr/bin/env python3
"""a classification head for transfer learning"""
from tensorflow import keras


def add_classification_head(base_model, num_classes):
    """adds a classification head to a base model"""
    x = base_model.output
    x = keras.layers.Dense(1024, activation='relu')(x)
    predictions = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=base_model.input, outputs=predictions)
    return model
