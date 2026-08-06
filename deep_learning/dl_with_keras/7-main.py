#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import os
import random
import numpy as np

build_model = __import__('0-sequential').build_model
compile_model = __import__('2-compile').compile_model
train_model = __import__('3-train').train_model
load_model = __import__('5-save_load_model').load_model
predict = __import__('7-predict').predict


def set_seed(SEED):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

set_seed(0)

_ , (x_test, y_test) = keras.datasets.mnist.load_data()

x_test = x_test.reshape(x_test.shape[0], -1)
x_test = x_test.astype('float32') / 255
y_test = keras.utils.to_categorical(y_test, num_classes=10)

model = load_model("mnist_model.keras")

predictions = predict(model, x_test, verbose=0)
print(predictions)
true_labels = np.argmax(y_test, axis=1).tolist()
for i in range(20):
    print(f"Example {i + 1}: Predicted = {predictions[i]}, True = {true_labels[i]}")