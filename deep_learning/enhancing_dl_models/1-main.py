#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import os
import random
import pandas as pd
import matplotlib.pyplot as plt

get_optimizer_SGD = __import__('1-momentum_sgd_variants').get_optimizer_SGD

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

optimizers = ['SGD', 'SGD+Momentum', 'SGD+Momentum+Nesterov']
learning_rate = 0.01
momentum = 0.9
batch_size = 32
input_dim = x_train.shape[1]
hidden_units = 4
epochs = 5

results = []
histories = {}

# Training loop
for opt_name in optimizers:
    print(f"\nTraining with {opt_name}...")

    optimizer = get_optimizer_SGD(opt_name, learning_rate, momentum, nesterov=True)
    model = build_model(input_dim, hidden_units)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        verbose=1
    )

    val_acc = history.history['val_accuracy'][-1]
    train_acc = history.history['accuracy'][-1]

    results.append([opt_name, val_acc, train_acc])
    histories[opt_name] = history

df = pd.DataFrame(results, columns=['Optimizer', 'Val Acc', 'Train Acc'])
print("\nPerformance Comparison of SGD Variants:\n")
print(df)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for name, history in histories.items():
    plt.plot(history.history['accuracy'], label=f'{name}')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
for name, history in histories.items():
    plt.plot(history.history['val_accuracy'], label=f'{name}')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig("task1.png")
plt.show()