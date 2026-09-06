#!/usr/bin/env python3
"""Builds, trains and saves an image classifier for the Caltech-101 dataset
using two-phase transfer learning (frozen head training + fine-tuning)."""
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = "101_ObjectCategories/101_ObjectCategories"
AUTOTUNE = tf.data.AUTOTUNE


def build_data_augmentation():
    """Builds a data augmentation pipeline as a Keras layer."""
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal", seed=42),
            keras.layers.RandomRotation(0.15, seed=42),
            keras.layers.RandomZoom(0.15, seed=42),
            keras.layers.RandomTranslation(0.1, 0.1, seed=42),
            keras.layers.RandomContrast(0.1, seed=42),
        ],
        name="data_augmentation",
    )


def build_feature_extractor():
    """Loads MobileNetV2  as a frozen feature extractor."""
    base_model = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False
    return base_model


def load_datasets():
    """Loads and prepares the train/validation datasets from disk."""
    train_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    val_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names, num_classes


def build_model(num_classes, data_augmentation, base_model):
    """Assembles: augmentation -> preprocessing -> base -> head."""
    preprocess_input = keras.applications.mobilenet_v2.preprocess_input

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


def train_transfer_model():
    """Builds, trains (head then fine-tune) and saves an image classifier."""

    # -----------------------------
    # 1. Load dataset
    # -----------------------------
    train_ds, val_ds, class_names, num_classes = load_datasets()
    print(f"Found {num_classes} classes.")

    # -----------------------------
    # 2. Build preprocessing / augmentation
    # -----------------------------
    data_augmentation = build_data_augmentation()

    # -----------------------------
    # 3. Load pretrained base model
    # -----------------------------
    base_model = build_feature_extractor()

    # -----------------------------
    # 4. Build full model
    # -----------------------------
    model = build_model(num_classes, data_augmentation, base_model)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_path = "best_head_model.keras"

    # -----------------------------
    # 5. Phase 1: train the classification head, base frozen
    # -----------------------------
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=4, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                checkpoint_path, monitor="val_accuracy",
                save_best_only=True, save_weights_only=False,
            ),
        ],
    )

    # -----------------------------
    # 6. Phase 2: fine-tune top layers of the base model
    #    MobileNetV2 has ~155 layers; unfreeze the last 55 for more capacity.
    # -----------------------------
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 55
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    finetune_checkpoint_path = "best_finetuned_model.keras"

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        verbose=1,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=5, restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                finetune_checkpoint_path, monitor="val_accuracy",
                save_best_only=True, save_weights_only=False,
            ),
        ],
    )

    # -----------------------------
    # 7. Report final validation accuracy and save final model
    # -----------------------------
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"Final validation accuracy: {val_acc:.4f}")

    model.save("caltech101_model.h5")

    return model


if __name__ == "__main__":
    train_transfer_model()
