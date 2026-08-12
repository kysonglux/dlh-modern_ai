#!/usr/bin/env python3
"""return complied keras model based on the specified weight initialization"""

from tensorflow import keras


def build_model_initializer_by_activation(input_dim, hidden_units, activation):
    """returns a compiled keras model
    based on the specified weight initialization method"""
    if activation in ("sigmoid", "tanh"):
        initializer = keras.initializers.GlorotUniform()
    elif activation in ("relu", "leaky_relu"):
        initializer = keras.initializers.HeNormal()
    else:
        raise ValueError("Invalid activation function")

    model = keras.Sequential([
        keras.layers.Dense(
            hidden_units,
            kernel_initializer=initializer,
            input_shape=(input_dim,)
        )
    ])

    if activation == "leaky_relu":
        model.add(keras.layers.LeakyReLU())
    else:
        model.add(keras.layers.Activation(activation))

    model.add(keras.layers.Dense(10, activation="softmax"))

    return model
