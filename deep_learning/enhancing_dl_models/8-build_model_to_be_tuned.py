#!/usr/bin/env python3
"""create a Keras model for multi-class classification"""
from tensorflow import keras


def build_model(hp):
    """create a Keras model for multi-class classification"""

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(784,)))
    num_layers = hp.Int('num_layers', min_value=1, max_value=2, step=1)
    units = hp.Int('units', min_value=4, max_value=12, step=4)
    activation = hp.Choice('activation', values=['relu', 'sigmoid'])

    model.add(keras.layers.Dense(units=units, activation=activation))
    for _ in range(num_layers - 1):
        model.add(keras.layers.Dense(units=units, activation=activation))
    model.add(keras.layers.Dense(10, activation='softmax'))
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
