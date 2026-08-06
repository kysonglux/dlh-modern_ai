#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import os
import random
import numpy as np

build_deep_model = __import__('8-deep_nn_model').build_deep_model
compile_model = __import__('2-compile').compile_model
train_model = __import__('3-train').train_model
evaluate_model = __import__('4-evaluate').evaluate_model
predict = __import__('7-predict').predict


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
hidden_layers = [20,15,10,5]

model = build_deep_model(input_dim, hidden_layers)
compile_model(model)
model.summary()

epochs = 200
train_model(model, x_train, y_train, epochs, verbose=1)

print("\n Evaluate Deep NN on test data")
results = evaluate_model(model, x_test, y_test)
print("test loss, test accuracy:", results)

print("\n Deep NN Predictions")
predictions = predict(model, x_test, verbose=0)
true_labels = np.argmax(y_test, axis=1).tolist()
for i in range(20):
    print(f"Example {i + 1}: Predicted = {predictions[i]}, True = {true_labels[i]}")