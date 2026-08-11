#!/usr/bin/env python3
"""returns the SGD optimizer configured with momentum and nesterov"""
from tensorflow import keras


def get_optimizer_SGD(name, lr, momentum=0.0, nesterov=False):
    """returns the SGD optimizer configured with momentum and nesterov"""
    if name == "SGD":
        return keras.optimizers.SGD(learning_rate=lr)
    elif name == "SGD+Momentum":
        return keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
    elif name == "SGD+Momentum+Nesterov":
        return keras.optimizers.SGD(learning_rate=lr,
                                    momentum=momentum, nesterov=nesterov)
    else:
        raise ValueError("Invalid optimizer name")
