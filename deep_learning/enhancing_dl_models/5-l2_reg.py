#!/usr/bin/env python3
"""create a keras model with l2 regularization"""


def build_model_with_L2_regularization(input_dim,
                                       hidden_units, n_layers, lambda_l2):
    """build a keras model with l2 regularization

    Args:
        input_dim (int): the input dimension
        hidden_units (int): the number of hidden units in each layer
        n_layers (int): the number of hidden layers
        lambda_l2 (float): the l2 regularization parameter

    Returns:
        keras.Model: the keras model
    """
    from tensorflow import keras

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(input_dim,)))

    for _ in range(n_layers):
        model.add(keras.layers.Dense
                  (hidden_units, activation='relu',
                   kernel_regularizer=keras.regularizers.l2(lambda_l2)))

    model.add(keras.layers.Dense(10, activation='softmax'))

    return model
