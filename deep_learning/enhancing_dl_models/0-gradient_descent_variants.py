#!/usr/bin/env python3
"""return gradient descent optimizer and batch size"""

from tensorflow import keras


def train_with_gradient_descent_variant(variant,
                                        learning_rate, x_train, batch_size):
    """returns the optimizer and batch size based
    on the gradient descent variant specified"""
    if variant == "batch":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        bs = x_train.shape[0]
        return optimizer, bs
    elif variant == "stochastic":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        bs = 1
        return optimizer, bs
    elif variant == "mini_batch":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        bs = batch_size
        return optimizer, bs
    else:
        raise ValueError("Invalid gradient descent variant")
