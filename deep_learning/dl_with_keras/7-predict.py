#!/usr/bin/env python3
"""predict using a trained model"""

import tensorflow as tf


def predict(model, X, verbose=0):
    """Predict using a trained model.

    Args:
        model (keras.Model): The trained Keras model.
        X (numpy.ndarray): Input data for prediction.
        verbose (int): Verbosity mode. 0 = silent, 1 = progress bar.
    """
    predictions = model.predict(X, verbose=verbose)
    return predictions
