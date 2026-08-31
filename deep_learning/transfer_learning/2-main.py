#!/usr/bin/env python3
"""
Main 2 — Fine-Tuning

Unfreezes the top layers of the pretrained backbone and fine-tunes
the model with a reduced learning rate.
"""

import tensorflow as tf
from tensorflow import keras
build_feature_extractor = __import__('0-frozen_extractor'
                                     ).build_feature_extractor
add_classification_head = __import__('1-classification_head'
                                     ).add_classification_head
unfreeze_top_layers = __import__('2-unfreeze_top'
                                 ).unfreeze_top_layers
tf.random.set_seed(42)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 2
EPOCHS_FINETUNE = 3
NUM_CLASSES = 10
UNFREEZE_LAYERS = 20


def prepare_datasets():
    (x_train, y_train), (x_val, y_val) = keras.datasets.cifar10.load_data()

    def preprocess(x, y):
        x = tf.image.resize(x, IMG_SIZE)
        x = tf.cast(x, tf.float32) / 255.0
        return x, tf.squeeze(y)

    train = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .map(preprocess)
        .shuffle(2000)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .map(preprocess)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train, val


train_ds, val_ds = prepare_datasets()

feature_extractor = build_feature_extractor()
model = add_classification_head(feature_extractor, NUM_CLASSES)

# Stage 1 — train head only
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD)

# Stage 2 — unfreeze top layers of MobileNetV2 backbone
mobilenet_backbone = feature_extractor.layers[1]
unfreeze_top_layers(mobilenet_backbone, UNFREEZE_LAYERS)

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE)