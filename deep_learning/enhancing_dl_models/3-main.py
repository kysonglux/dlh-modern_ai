#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import os
import random
import matplotlib.pyplot as plt

get_optimizer_SGD_with_schedule = __import__('3-learning_rate_schedule').get_optimizer_SGD_with_schedule

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

initial_lr = 0.1
decay_steps = 1000
decay_rate = 0.96
momentum = 0.9
batch_size = 64
epochs = 35
input_dim = x_train.shape[1]
hidden_units = 4

schedules = ['exponential', 'inverse_time']
histories = {}
lr_values = {}

for schedule in schedules:
    print(f"\nTraining with {schedule} decay schedule...")
    optimizer, lr_schedule = get_optimizer_SGD_with_schedule(schedule, initial_lr, decay_steps, decay_rate, momentum)
    print(type(lr_schedule))
    model = build_model(input_dim, hidden_units)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        verbose=1
    )

    histories[schedule] = history
    lr_values[schedule] = [lr_schedule(epoch * (len(x_train) // batch_size)).numpy() for epoch in range(epochs)]

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
for schedule in schedules:
    plt.plot(histories[schedule].history['accuracy'], label=f"{schedule} train")
plt.title("Training & Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
for schedule in schedules:
    plt.plot(lr_values[schedule], label=f"{schedule}")
plt.title("Learning Rate over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Learning Rate")
plt.legend()
plt.tight_layout()
plt.savefig("task3.png")
plt.show()