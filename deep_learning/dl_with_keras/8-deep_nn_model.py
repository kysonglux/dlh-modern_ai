#!/usr/bin/env python3
"""create deep neural network to perform multi-class classification"""
from tensorflow import keras


def build_deep_model(input_dim, hidden_layers):
    """Create a deep neural network.

    Args:
        input_dim (int): Flattened input dimensionality.
        hidden_layers (list): A list containing the number of units.
    """
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    for neurons in hidden_layers:
        model.add(keras.layers.Dense(neurons, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    return model
