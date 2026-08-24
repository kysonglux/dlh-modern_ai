#!/usr/bin/env python3
"""creates a convolutional neural network in keras"""
from tensorflow import keras


def create_cnn_model(input_shape, filters, kernel_sizes, activations,
                     pooling_type='max'):
    """creates a convolutional neural network in keras

    Args:
        input_shape (tuple): shape of the input data
        filters (list): list containing the number of filters for each
                        convolutional layer
        kernel_sizes (list): list containing the kernel size for each
                            convolutional layer
        activations (list): list containing the activation functions for each
                            convolutional layer
        pooling_type (str): type of pooling to use ('max' or 'avg')"""

    if pooling_type not in ['max', 'avg']:
        raise ValueError("pooling_type must be 'max' or 'avg'")

    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=filters[0],
                                  kernel_size=kernel_sizes[0],
                                  activation=activations[0],
                                  padding='valid',
                                  input_shape=input_shape))

    if pooling_type == 'max':
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    else:
        model.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(filters=filters[1],
                                  kernel_size=kernel_sizes[1],
                                  activation=activations[1],
                                  padding='valid'))
    if pooling_type == 'max':
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    else:
        model.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=10, activation='softmax'))

    return model
