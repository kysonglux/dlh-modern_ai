#!/usr/bin/env python3
"""
Task 3: Data Augmentation (CIFAR-10)
"""

from tensorflow import keras

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

data_augmentation = __import__('3-data_aug').build_data_augmentation()
feature_extractor = __import__('0-frozen_extractor').build_feature_extractor()

# Build model
inputs = keras.Input(shape=(32, 32, 3))

# Resize FIRST
x = keras.layers.Resizing(224, 224)(inputs)

# Augmentation only affects training
x = data_augmentation(x)

# Feature extraction
x = feature_extractor(x)

# Classification head
x = keras.layers.Dense(128, activation="relu")(x)
outputs = keras.layers.Dense(10, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=64
)

model.evaluate(x_test, y_test)