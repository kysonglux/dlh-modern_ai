#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras
build_feature_extractor = __import__('0-frozen_extractor'
                                     ).build_feature_extractor
add_classification_head = __import__('1-classification_head'
                                     ).add_classification_head

tf.random.set_seed(42)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 3
NUM_CLASSES = 10


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

base_model = build_feature_extractor()
model = add_classification_head(base_model, NUM_CLASSES)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)