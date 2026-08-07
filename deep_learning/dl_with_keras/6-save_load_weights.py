#!/usr/bin/env python3
"""save and load model weights"""


def save_model_weights(model, filepath):
    """Save the weights of a Keras model to a file.

    Args:
        model (keras.Model): The Keras model whose weights are to be saved.
        filepath (str): The path to the file where the weights will be saved.
    """
    model.save_weights(filepath)


def load_model_weights(model, filepath):
    """Load the weights of a Keras model from a file.

    Args:
        model (keras.Model): The Keras model whose weights are to be loaded.
        filepath (str): The path to the file from which the weights
        will be loaded.
    """
    model.load_weights(filepath)
