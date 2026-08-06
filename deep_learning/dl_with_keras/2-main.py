#!/usr/bin/env python3

from tensorflow import keras
build_model = __import__('0-sequential').build_model
compile_model = __import__('2-compile').compile_model


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

print("Model Loss Function: " ,model.loss)
print("Model Optimizer: " ,model.optimizer.__class__)
print("Learning Rate:"  ,model.optimizer.learning_rate.numpy())