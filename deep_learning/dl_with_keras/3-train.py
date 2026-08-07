#!/usr/bin/env python3
"""train a keras model"""


def train_model(model, X, Y, epochs, verbose=1):
    """Train a model using mini-batch gradient descent.

    Args:
        model (keras.Model): The Keras model to train.
        X (numpy.ndarray): The input data for training.
        Y (numpy.ndarray): The target data for training.
        epochs (int): The number of epochs to train the model.
        verbose (int): Verbosity mode.
        0 = silent, 1 = progress bar, 2 = one line per epoch.
    """
    model.fit(X, Y, epochs=epochs, verbose=verbose)
