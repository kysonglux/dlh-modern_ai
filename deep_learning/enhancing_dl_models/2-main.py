#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import os
import random
import matplotlib.pyplot as plt

get_optimizer = __import__('2-adaptive_optimizers').get_optimizer

def set_seed(SEED):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

set_seed(0)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, num_classes=10)

def build_model(input_dim, hidden_units):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    model.add(keras.layers.Dense(hidden_units, activation='sigmoid'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

# Training settings
optimizers = ['sgd', 'adam', 'rmsprop']
learning_rate = 0.01
momentum = 0.9
beta_1 = 0.9
beta_2 = 0.999
rho = 0.9

hidden_units = 4
val_split = 0.2
batch_size = 64
epochs = 10
input_dim = x_train.shape[1]
hidden_units = 4
histories = {}

for opt_name in optimizers:
    print(f"\nTraining with {opt_name} optimizer")
    optimizer = get_optimizer(opt_name, learning_rate, momentum, beta_1, beta_2, rho)
    print(optimizer.get_config())
    print("\n")
    model = build_model(input_dim, hidden_units)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        verbose=1
    )

    histories[opt_name] = history

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
for name in optimizers:
    plt.plot(histories[name].history['accuracy'], label=f"{name} train")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
for name in optimizers:
    plt.plot(histories[name].history['val_accuracy'], label=f"{name} val")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("task2.png")
plt.show()