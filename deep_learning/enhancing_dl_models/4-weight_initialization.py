#!/usr/bin/env python3
"""return complied keras model based on the specified weight initialization"""

from tensorflow import keras


def build_model_initializer_by_activation(input_dim, hidden_units, activation):
    """returns a compiled keras model
    based on the specified weight initialization method"""
    if activation == "sigmoid":
        initializer = keras.initializers.GlorotUniform()
        activation_layer = keras.layers.Activation("sigmoid")
    elif activation == "tanh":
        initializer = keras.initializers.GlorotUniform()
        activation_layer = keras.layers.Activation("tanh")
    elif activation == "relu":
        initializer = keras.initializers.HeNormal()
        activation_layer = keras.layers.Activation("relu")
    elif activation == "leaky_relu":
        initializer = keras.initializers.HeNormal()
        activation_layer = keras.layers.Activation("leaky_relu")
    else:
        raise ValueError("Invalid activation function")

    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(hidden_units,
                           kernel_initializer=initializer,),
        activation_layer,
        keras.layers.Dense(hidden_units,
                           kernel_initializer=initializer,),
        activation_layer,
        keras.layers.Dense(10, activation='softmax')
    ])

    return model
