#!/usr/bin/env python3
"""returns keras optimizer based on adaptive optimizer"""
from tensorflow import keras


def get_optimizer(name, learning_rate, momentum, beta_1, beta_2, rho):
    """returns keras optimizer based
    on the specified adaptive optimizer"""
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=learning_rate,
                                    momentum=momentum)
    elif name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=learning_rate, rho=rho)
    elif name == "adam":
        return keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
        )
    else:
        raise ValueError("Invalid optimizer name")
