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

    inputs = keras.Input(shape=input_shape)
    x = inputs

    for i in range(len(filters)):
        x = keras.layers.Conv2D(filters=filters[i],
                                kernel_size=kernel_sizes[i],
                                activation=activations[i],
                                padding='valid')(x)
        if pooling_type == 'max':
            x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        else:
            x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)

    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
