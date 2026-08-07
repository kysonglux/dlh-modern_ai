#!/usr/bin/env python3
"""assess model performance"""


def evaluate_model(model, X, Y, verbose=0):
    """Evaluate the model on the given data.

    Args:
        model (keras.Model): The Keras model to evaluate.
        X (numpy.ndarray): The input data for evaluation.
        Y (numpy.ndarray): The target data for evaluation.

    Returns:
        float: The loss value.
        float: The accuracy value.
    """
    loss, accuracy = model.evaluate(X, Y, verbose=0)
    return loss, accuracy
