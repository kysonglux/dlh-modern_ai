#!/usr/bin/env python3
build_resnet101 = __import__('3-resnet_101').build_resnet101

model1 = build_resnet101()
model1.summary()