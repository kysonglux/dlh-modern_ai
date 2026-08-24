#!/usr/bin/env python3
"""train a convolutional neural network in tensorflow"""
from tensorflow import keras


def compile_and_train_cnn(model, epochs, batch_size, optimizer_name='adam',
                          optimizer_params=None, **kwargs):
    """compiles and trains a model using the adam optimization algorithm

    Args:
        model: the model to train
        epochs: the number of epochs to train for
        batch_size: the batch size for training
        optimizer_name: the type of optimizer to use (default is 'adam')
        optimizer_params: a dictionary of parameters for the optimizer

    Returns:
        The History object generated after training the model
    """
    if optimizer_params is None:
        optimizer_params = {}

    x_train = kwargs.get('x_train')
    y_train = kwargs.get('y_train')
    x_val = kwargs.get('x_val')
    y_val = kwargs.get('y_val')

    if x_train is None or y_train is None:
        raise ValueError("x_train and y_train must be provided"
                         "as keyword arguments.")
    if x_val is None or y_val is None:
        raise ValueError("x_val and y_val must be provided"
                         "as keyword arguments.")

    optimizer = keras.optimizers.get({
        "class_name": optimizer_name,
        "config": optimizer_params})

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        verbose=2
    )

    return model, history
