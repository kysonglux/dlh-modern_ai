#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import random
import matplotlib.pyplot as plt

get_early_stopping_callback = __import__('7-early_stopping').get_early_stopping_callback

def set_seed(SEED):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

set_seed(0)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, 10)


def build_model(input_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

input_dim = x_train.shape[1]
batch_size = 128
epochs = 100

model = build_model(input_dim)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping_callback = get_early_stopping_callback(
    patience=3,
    monitor='val_loss',
    verbose=1
)

history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    callbacks=[early_stopping_callback]
)

train_error = history.history['loss']
val_error = history.history['val_loss']
epochs = range(1, len(train_error) + 1)

plt.figure(figsize=(8, 6))
plt.plot(epochs, train_error, label='Training Loss', marker='o')
plt.plot(epochs, val_error, label='Validation Loss', marker='o')

best_epoch = np.argmin(val_error) + 1
plt.axvline(x=best_epoch, color='green', linestyle='--', label=f'Best Weights Saved (Epoch {best_epoch})')
plt.fill_between(epochs, train_error, val_error, color='gray', alpha=0.2)

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Balancing Learning and Overfitting with Early Stopping')
plt.legend()
plt.savefig("task7.png")
plt.show()
