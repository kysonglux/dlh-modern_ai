#!/usr/bin/env python3
"""create a customizable early stopping callback for keras"""
from tensorflow import keras


def get_early_stopping_callback(patience, monitor='val_loss', verbose=1):
    """create a customizable early stopping callback for keras"""

    return keras.callbacks.EarlyStopping(
        patience=patience,
        monitor=monitor,
        verbose=verbose
    )
