#!/usr/bin/env python3

mobilenet = __import__('6-mobilenetv1').mobilenet

model = mobilenet()
model.summary()