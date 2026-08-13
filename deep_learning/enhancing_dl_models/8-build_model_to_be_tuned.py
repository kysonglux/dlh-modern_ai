#!/usr/bin/env python3
"""create a Keras model for multi-class classification"""
from tensorflow import keras


def build_model(hp):
    """create a Keras model for multi-class classification"""

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(784,)))
    model.add(keras.layers.Dense(
        units=hp.Int('units', min_value=4, max_value=12, step=4),
        activation=hp.Choice('activation', values=['relu', 'sigmoid'])))

    for i in range(hp.Int('num_layers', 1, 2)):
        model.add(keras.layers.Dense(
            units=hp.Int('units', min_value=4, max_value=12, step=4),
            activation=hp.Choice('activation', values=['relu', 'sigmoid'])))

    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3])),
        metrics=['accuracy']
    )
    return model
