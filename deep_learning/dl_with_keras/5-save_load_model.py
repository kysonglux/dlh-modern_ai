#!/usr/bin/env python3
"""save and load model"""
from tensorflow import keras


def save_model(network, filename):
    """Save a model's architecture and weights.

    Args:
        network (keras.Model): The Keras model to save.
        filename (str): The path where the model will be saved.
    """
    network.save(filename)


def load_model(filename):
    """Load a model from a file.

    Args:
        filename (str): The path to the saved model file.
    """
    return keras.models.load_model(filename)
