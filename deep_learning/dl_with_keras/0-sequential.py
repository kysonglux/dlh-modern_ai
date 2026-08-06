#!/usr/bin/env python3
"""create shallow neural network """

from tensorflow import keras


def build_model(input_dim, n_h=512):
    """Create a shallow neural network.

    Args:
        input_dim (int): Flattened input dimensionality.
        n_h (int): Number of units in the hidden layer.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(n_h, activation="sigmoid"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    return model
