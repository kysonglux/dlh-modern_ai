#!/usr/bin/env python3
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

create_cnn_model = __import__('0-create_cnn_model').create_cnn_model
compile_and_train_cnn = __import__('1-train_cnn').compile_and_train_cnn

SEED = 0
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

x_train, x_val = x_train[:50000], x_train[50000:]
y_train, y_val = y_train[:50000], y_train[50000:]

filters = [32, 64]
kernels = [(3, 3), (3, 3)]

optimizers = {
    "Adam": {"name": "adam", "params": {}},
    "SGD": {"name": "sgd", "params": {"learning_rate": 0.01}},
    "RMSprop": {"name": "rmsprop", "params": {}}
}

results = []

for opt_name, opt_cfg in optimizers.items():
    print(f"\nTraining with {opt_name}")

    model = create_cnn_model(
        input_shape=(28, 28, 1),
        filters=filters,
        kernel_sizes=kernels,
        activations=['relu', 'relu'],
        pooling_type='max'
    )

    model, history = compile_and_train_cnn(
        model=model,
        epochs=5,
        batch_size=64,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        optimizer_name=opt_cfg["name"],
        optimizer_params=opt_cfg["params"]
    )