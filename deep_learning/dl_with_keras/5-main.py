#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import os
import random
import numpy as np

build_model = __import__('0-sequential').build_model
compile_model = __import__('2-compile').compile_model
train_model = __import__('3-train').train_model
evaluate_model = __import__('4-evaluate').evaluate_model
save_model = __import__('5-save_load_model').save_model
load_model = __import__('5-save_load_model').load_model



def set_seed(SEED):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

set_seed(0)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

input_dim = x_train.shape[1]
n_h = 4

model = build_model(input_dim, n_h)
compile_model(model)

epochs = 200
train_model(model, x_train, y_train, epochs, verbose=1)

print(model.get_weights())
print(model.optimizer.__class__)

print("\n ######### Save Model #########")
save_model(model, "mnist_model.keras")
print("Model saved")

print("\n ######### Load Model #########")
loaded_model = load_model("mnist_model.keras")
loaded_model.summary()
print(loaded_model.get_weights())
print(loaded_model.optimizer.__class__)

print("\n Evaluate the loaded model")
results = evaluate_model(loaded_model,x_test, y_test)
print("test loss, test accuracy:", results)