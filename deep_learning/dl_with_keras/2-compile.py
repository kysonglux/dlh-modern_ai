#!/usr/bin/env python3
"""compile model and train it"""

from pyexpat import model

from tensorflow import keras


def compile_model(model, learning_rate=0.01):
    """Compile and train the model.

    Args:
        model (keras.Model): The Keras model to compile and train.
        learning_rate (float): The learning rate for the optimizer.
    """

    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
