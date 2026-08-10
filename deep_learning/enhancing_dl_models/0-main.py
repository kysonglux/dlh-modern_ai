#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import time
import pandas as pd
import os
import random

train_with_gradient_descent_variant = __import__('0-gradient_descent_variants').train_with_gradient_descent_variant


def set_seed(SEED):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

set_seed(0)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1)
x_train = x_train.astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, num_classes=10)

def build_model(input_dim, hidden_units):
    """
    Builds a simple feedforward model.
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    model.add(keras.layers.Dense(hidden_units, activation='sigmoid'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

learning_rate = 0.01
mini_batch_size = 32
input_dim = x_train.shape[1]
hidden_units = 4
epochs = 10

results = []

for variant in ['batch', 'mini_batch', 'stochastic']:
    print(f"\nTraining with {variant} gradient descent...")

    optimizer, bs = train_with_gradient_descent_variant(
        variant, learning_rate, x_train, mini_batch_size
    )

    model = build_model(input_dim, hidden_units)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        batch_size=bs,
        epochs=epochs,
        validation_split=0.2,
        verbose=1
    )
    end_time = time.time()

    val_acc = history.history['val_accuracy'][-1]
    train_acc = history.history['accuracy'][-1]
    train_time = end_time - start_time

    results.append([variant, val_acc, train_acc, train_time])


df = pd.DataFrame(results, columns=['Variant', 'Val Acc', 'Train Acc', 'Train Time (s)'])
print("\nComparison of Gradient Descent Variants:\n")
print(df)