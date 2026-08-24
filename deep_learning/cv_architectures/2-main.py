#!/usr/bin/env python3

from tensorflow import keras as K
bottleneck_block = __import__('2-bottleneck_block').bottleneck_block

X = K.Input(shape=(28, 28, 64))
Y = bottleneck_block(
    X,
    filters=16,
    stride=1,
    downsample=False,
    name="bottleneck"
)
model = K.Model(inputs=X, outputs=Y)
model.summary()