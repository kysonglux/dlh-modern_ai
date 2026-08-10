#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import random
import matplotlib.pyplot as plt

build_model_with_L2_regularization = __import__('5-l2_reg').build_model_with_L2_regularization

def set_seed(SEED):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

set_seed(0)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, num_classes=10)

input_dim = x_train.shape[1]
hidden_units = 64
batch_size = 128
epochs = 15
n_layers = 3

print("Training without regularization")
model_no_reg = build_model_with_L2_regularization(input_dim, hidden_units, n_layers, lambda_l2=0)
model_no_reg.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_no_reg = model_no_reg.fit(
    x_train, y_train,
    validation_split=0.2,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1)

print("\nTraining with L2 regularization")
model_l2 = build_model_with_L2_regularization(input_dim, hidden_units, n_layers, lambda_l2=1e-6)
model_l2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_l2 = model_l2.fit(
    x_train, y_train,
    validation_split=0.2,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1)

def get_weights_from_model(model):
    weights = []
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            # kernel weights
            weights.append(layer.get_weights()[0].flatten())
    if weights:
        return np.concatenate(weights)
    return np.array([])

weights_no_reg = get_weights_from_model(model_no_reg)
weights_l2 = get_weights_from_model(model_l2)

plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
plt.hist(weights_no_reg, bins=50, alpha=0.7)
plt.title("Weight Distribution WITHOUT L2 Regularization")
plt.xlabel("Weight values")
plt.ylabel("Frequency")

plt.subplot(1,2,2)
plt.hist(weights_l2, bins=50, color='orange', alpha=0.7)
plt.title("Weight Distribution WITH L2 Regularization")
plt.xlabel("Weight values")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("task5.png")
plt.show()
