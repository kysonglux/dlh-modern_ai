#!/usr/bin/env python3
"""create shallow neural network without using Sequential API"""


from tensorflow import keras


def build_model(input_dim, neurons_h):
    """Create a shallow neural network.

    Args:
        input_dim (int): Flattened input dimensionality.
        neurons_h (int): Number of units in the hidden layer.
    """
    input_layer = keras.layers.Input(shape=(input_dim,))
    hidden_layer = keras.layers.Dense(neurons_h,
                                      activation="sigmoid")(input_layer)
    output_layer = keras.layers.Dense(10, activation="softmax")(hidden_layer)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model
