#!/usr/bin/env python3
"""return complied keras model based on the specified weight initialization"""

from tensorflow import keras


def build_model_initializer_by_activation(input_dim, hidden_units, activation):
    """returns a compiled keras model
    based on the specified weight initialization method"""
    if activation == "sigmoid":
        initializer = keras.initializers.GlorotUniform()
    elif activation == "tanh":
        initializer = keras.initializers.GlorotUniform()
    elif activation == "relu":
        initializer = keras.initializers.HeNormal()
    elif activation == "leaky_relu":
        initializer = keras.initializers.HeNormal()
    else:
        raise ValueError("Invalid activation function")

    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(hidden_units,
                           activation=activation,
                           kernel_initializer=initializer,),
        keras.layers.Dense(10, activation='softmax')
    ])

    return model
