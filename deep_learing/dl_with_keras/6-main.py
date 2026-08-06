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
save_model_weights = __import__('6-save_load_weights').save_model_weights
load_model_weights = __import__('6-save_load_weights').load_model_weights



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

print("\n ######### Save Weights #########")
save_model_weights(model,"model_trained.weights.h5")
print("Model weights are saved")

print("\n ######### Load Weights #########")
new_model = build_model(input_dim, n_h)
compile_model(new_model)
load_model_weights(new_model, "model_trained.weights.h5")
print("Weights loaded into a new model instance\n")
new_model.summary()
print("\n Loaded model weights")
print(new_model.get_weights())

print("\n Evaluate the new model with loaded weights")
results = evaluate_model(new_model, x_test, y_test, verbose=0)
print("test loss, test accuracy:", results)