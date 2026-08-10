#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import random
import matplotlib.pyplot as plt

build_model_initializer_by_activation = __import__('4-weight_initialization').build_model_initializer_by_activation

def set_seed(SEED):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

set_seed(0)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, num_classes=10)

activations = ['sigmoid', 'tanh', 'relu', 'leaky_relu']
histories = {}
input_dim = x_train.shape[1]
hidden_units = 4
epochs = 15
batch_size = 128

for act in activations:
    print(f"\nTraining with activation: {act}")
    model = build_model_initializer_by_activation(input_dim, hidden_units, act)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    histories[act] = history

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for act in activations:
    plt.plot(histories[act].history['accuracy'], label=f'{act}')
plt.xlabel("Epochs")
plt.ylabel("Training Accuracy")
plt.title("Training Accuracy per Activation")
plt.legend()

plt.subplot(1, 2, 2)
for act in activations:
    plt.plot(histories[act].history['val_accuracy'], label=f'{act}')
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy per Activation")
plt.legend()

plt.tight_layout()
plt.savefig("task4.png")
plt.show()
