#!/usr/bin/env python3

from tensorflow import keras as K
depthwise_separable_conv = __import__('4-depthwise_separable_conv'
                                      ).depthwise_separable_conv

X = K.Input(shape=(32, 32, 64))
Y = depthwise_separable_conv(X, filters=128, stride=1)

model = K.Model(inputs=X, outputs=Y)
model.summary()