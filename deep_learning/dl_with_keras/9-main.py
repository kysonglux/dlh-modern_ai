#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import os
import random
import numpy as np

build_deep_model = __import__('8-deep_nn_model').build_deep_model
compile_model = __import__('2-compile').compile_model
log_to_tensorboard = __import__('9-tensorboard').log_to_tensorboard


def set_seed(SEED):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

set_seed(0)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1)
x_train = x_train.astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, num_classes=10)

input_dim = x_train.shape[1]
hidden_layers = [20,15,10,5]

model = build_deep_model(input_dim, hidden_layers)
compile_model(model)

epochs = 100
log_dir = "logs/tensorboard_demo"
log_to_tensorboard(log_dir, model, x_train, y_train, epochs, verbose=1)