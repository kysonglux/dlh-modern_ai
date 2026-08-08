#!/usr/bin/env python3
"""log a keras model to tensorboard"""
from tensorflow import keras
import datetime


def log_to_tensorboard(log_dir, model, X, Y, epochs, verbose=1):
    """Log a Keras model to TensorBoard.

    Args:
        model (keras.Model): The Keras model to log.
        log_dir (str): Directory where the TensorBoard logs will be saved.
        If None, a default directory with a timestamp will be created.
    """
    if log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"logs/fit/{timestamp}"

    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True)

    model.fit(X, Y, epochs=epochs, verbose=verbose, callbacks=[tensorboard_cb])

    return None
