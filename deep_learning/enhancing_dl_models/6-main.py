#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import os
import random
import matplotlib.pyplot as plt

build_model_with_dropout = __import__('6-dropout').build_model_with_dropout

def set_seed(SEED):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

set_seed(0)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, 10)

input_dim = x_train.shape[1]
hidden_units = 512
n_layers = 2
batch_size = 128
epochs = 20
dropout_rate_input = 0.2
dropout_rate_hidden = 0.5

histories = {}

print("Training without dropout...")
model_no_dropout = build_model_with_dropout(input_dim, hidden_units, n_layers, dropout_rate_input=0, dropout_rate_hidden=0)
model_no_dropout.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist_no_dropout = model_no_dropout.fit(
    x_train, y_train,
    validation_split=0.2,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1
)
histories['No Dropout'] = hist_no_dropout

print("\nTraining with dropout...")
model_dropout = build_model_with_dropout(input_dim, hidden_units, n_layers, dropout_rate_input, dropout_rate_hidden)
model_dropout.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist_dropout = model_dropout.fit(
    x_train, y_train,
    validation_split=0.2,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1
)
histories['With Dropout'] = hist_dropout

plt.figure(figsize=(12, 5))
epochs_range = range(1, epochs + 1)

plt.subplot(1, 2, 1)
train_acc_no_dropout = histories['No Dropout'].history['accuracy']
val_acc_no_dropout = histories['No Dropout'].history['val_accuracy']
plt.plot(epochs_range, train_acc_no_dropout, label='Train Accuracy')
plt.plot(epochs_range, val_acc_no_dropout, label='Validation Accuracy')
plt.fill_between(epochs_range, train_acc_no_dropout, val_acc_no_dropout, color='gray', alpha=0.2)
plt.title("Train vs Validation Accuracy (No Dropout)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
train_acc_dropout = histories['With Dropout'].history['accuracy']
val_acc_dropout = histories['With Dropout'].history['val_accuracy']
plt.plot(epochs_range, train_acc_dropout, label='Train Accuracy')
plt.plot(epochs_range, val_acc_dropout, label='Validation Accuracy')
plt.fill_between(epochs_range, train_acc_dropout, val_acc_dropout, color='gray', alpha=0.2)
plt.title("Train vs Validation Accuracy (With Dropout)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("task6.png")
plt.show()