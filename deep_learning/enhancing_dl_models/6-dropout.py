#!/usr/bin/env python3
"""create a keras model with dropout regularization"""
from tensorflow import keras


def build_model_with_dropout(input_dim, hidden_units, n_layers,
                             dropout_rate_input, dropout_rate_hidden):
    """builds a keras model with dropout regularization"""

    model = keras.Sequential()
    model.add(keras.layers.Dense(hidden_units, activation='relu',
                                 input_shape=(input_dim,)))
    model.add(keras.layers.Dropout(dropout_rate_input))
    for _ in range(n_layers):
        model.add(keras.layers.Dense(hidden_units, activation='relu'))
        model.add(keras.layers.Dropout(dropout_rate_hidden))
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model
