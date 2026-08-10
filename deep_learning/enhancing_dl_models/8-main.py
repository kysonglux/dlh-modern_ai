#!/usr/bin/env python3

from keras_tuner import HyperParameters

build_model = __import__('8-build_model_to_be_tuned').build_model

hp = HyperParameters()
model = build_model(hp)

print(model.input_shape)
print(type(model.optimizer).__name__)
print(model.layers[-1].activation.__name__)
