#!/usr/bin/env python3

from tensorflow import keras as K
mobilenet_backbone = __import__('5-mobilenet_backbone'
                                ).mobilenet_backbone

X1 = K.Input(shape=(224, 224, 3))
Y1 = mobilenet_backbone(X1)

model1 = K.Model(inputs=X1, outputs=Y1)
model1.summary()