#!/usr/bin/env python3
"""Transfer learning from Stanford Cars to 101 Object Categories

Source dataset:
    Stanford Cars
    8,144 training images
    196 classes

Target dataset:
    101_objectCategories
    9,145 images
    102 classes

Training:
    1. ImageNet -> Stanford Cars
    2. Stanford Cars -> 101 Object Categories

Final output:
    caltech101_model.h5
"""

import os
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from tensorflow import keras


# ============================================================
# 0. Configuration
# ============================================================

tf.random.set_seed(42)
np.random.seed(42)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ------------------------------------------------------------
# Stanford Cars
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARS_TRAIN_DIR = os.path.join(BASE_DIR, "archive", "cars_train", "cars_train")
CARS_DEVKIT_DIR = os.path.join(BASE_DIR, "archive", "car_devkit", "devkit")
CARS_ANNO_FILE = os.path.join(CARS_DEVKIT_DIR, "cars_train_annos.mat")

# ------------------------------------------------------------
# Target dataset
# ------------------------------------------------------------

TARGET_DIR = os.path.join(BASE_DIR, "101_objectCategories",
                          "101_objectCategories")

EXPECTED_TARGET_CLASSES = 102

# ------------------------------------------------------------
# Output files
# ------------------------------------------------------------

STANFORD_MODEL_FILE = "stanford_cars_model.keras"
FINAL_MODEL_FILE = "caltech101_model.h5"


# ============================================================
# 1. Data augmentation
# ============================================================

def build_data_augmentation():
    """Build common image augmentation layers."""

    return keras.Sequential([
            keras.layers.RandomFlip("horizontal", seed=42),
            keras.layers.RandomRotation(0.10, seed=42),
            keras.layers.RandomZoom(0.10, seed=42),
            keras.layers.RandomContrast(0.10, seed=42),
        ], name="data_augmentation")

# ============================================================
# 2. Read Stanford Cars .mat annotations
# ============================================================


def load_stanford_cars_annotations():
    """Read cars_train_annos.mat.

    The Stanford Cars annotations contain:
        fname
        bbox_x1
        bbox_y1
        bbox_x2
        bbox_y2
        class

    We only need:
        fname
        class
    """

    print("\n========================================")
    print("Loading Stanford Cars annotations")
    print("========================================")

    if not os.path.exists(CARS_ANNO_FILE):
        raise FileNotFoundError(
            f"Cannot find:\n{CARS_ANNO_FILE}\n\n"
            "Check that cars_train_annos.mat is inside "
            "the devkit directory."
        )

    mat = loadmat(CARS_ANNO_FILE)

    annotations = mat["annotations"]

    image_paths = []
    labels = []

    for annotation in annotations[0]:

        # Filename
        filename = str(annotation["fname"][0])

        # Class labels in Stanford Cars are 1-based.
        # TensorFlow uses 0-based labels.
        class_id = int(annotation["class"][0][0])

        label = class_id - 1

        image_path = os.path.join(CARS_TRAIN_DIR, filename)

        image_paths.append(image_path)
        labels.append(label)

    image_paths = np.array(image_paths)

    labels = np.array(labels, dtype=np.int32)

    print("Stanford Cars images found:", len(image_paths))

    print("Number of classes:", len(np.unique(labels)))

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    if len(image_paths) != 8144:
        print("\nWARNING:"f" Expected 8144 training images, "
              f"but found {len(image_paths)}.")

    if len(np.unique(labels)) != 196:
        print("\nWARNING:"f" Expected 196 classes, "
              f"but found {len(np.unique(labels))}.")

    # Check a few image files.
    missing = [path for path in image_paths[:20]
               if not os.path.exists(path)]

    if missing:
        raise FileNotFoundError(
            "Stanford Cars image files could not be found.\n"
            f"Example missing file: {missing[0]}\n\n"
            "Check CARS_TRAIN_DIR.")

    return image_paths, labels

# ============================================================
# 3. Load image from filename
# ============================================================


def load_image(path, label):
    """Read and resize an image."""

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, IMG_SIZE)

    image = tf.cast(image, tf.float32)

    return image, label


# ============================================================
# 4. Create Stanford Cars tf.data dataset
# ============================================================

def build_stanford_dataset(image_paths, labels, training=False):
    """Create TensorFlow dataset from image paths and labels."""

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels))

    if training:

        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=42,
            reshuffle_each_iteration=True)

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# ============================================================
# 5. Build MobileNetV2
# ============================================================


def build_backbone():
    """
    Build MobileNetV2 without the ImageNet classification head.

    ImageNet weights provide the initial representation.
    We then train this network on Stanford Cars.
    """

    base_model = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False, weights="imagenet")

    return base_model


# ============================================================
# 6. Train Stanford Cars model
# ============================================================

def train_stanford_cars():
    """
    Train MobileNetV2 on Stanford Cars.

    Result:
        Stanford-Cars-trained MobileNetV2 backbone.
    """

    print("\n")
    print("########################################")
    print("# STEP 1: TRAIN ON STANFORD CARS")
    print("########################################")

    # --------------------------------------------------------
    # Load annotations
    # --------------------------------------------------------

    image_paths, labels = (load_stanford_cars_annotations())

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    indices = np.arange(len(image_paths))

    np.random.shuffle(indices)

    split_index = int(0.8 * len(indices))

    train_indices = indices[:split_index]

    val_indices = indices[split_index:]

    train_paths = image_paths[train_indices]

    train_labels = labels[train_indices]

    val_paths = image_paths[val_indices]

    val_labels = labels[val_indices]

    print("\nStanford Cars training:", len(train_paths))

    print("Stanford Cars validation:", len(val_paths))

    # --------------------------------------------------------
    # TensorFlow datasets
    # --------------------------------------------------------

    train_ds = build_stanford_dataset(
        train_paths, train_labels, training=True)

    val_ds = build_stanford_dataset(
        val_paths, val_labels, training=False)

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    augmentation = build_data_augmentation()

    preprocess_input = (
        keras.applications.mobilenet_v2
        .preprocess_input)

    # --------------------------------------------------------
    # Backbone
    # --------------------------------------------------------

    base_model = build_backbone()

    # Freeze MobileNetV2 initially.
    base_model.trainable = False

    # --------------------------------------------------------
    # Stanford Cars classifier
    # --------------------------------------------------------

    inputs = keras.Input(
        shape=IMG_SIZE + (3,), name="input")

    x = augmentation(inputs)

    x = preprocess_input(x)

    x = base_model(x, training=False)

    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(256, activation="relu")(x)

    x = keras.layers.Dropout(0.3)(x)

    outputs = keras.layers.Dense(
        196, activation="softmax", name="cars_classifier")(x)

    cars_model = keras.Model(inputs, outputs)

    # ========================================================
    # Stanford Cars Phase 1
    # ========================================================

    print("\n")
    print("----------------------------------------")
    print("Stanford Cars Phase 1")
    print("Training classification head")
    print("----------------------------------------")

    cars_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])

    cars_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=3,
                restore_best_weights=True),

            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=2,
                min_lr=1e-6)
        ]
    )

    # ========================================================
    # Stanford Cars Phase 2
    # ========================================================

    print("\n")
    print("----------------------------------------")
    print("Stanford Cars Phase 2")
    print("Fine-tuning MobileNetV2")
    print("----------------------------------------")

    base_model.trainable = True

    # Freeze all but last 40 layers.
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    # Keep BatchNormalization layers frozen.
    for layer in base_model.layers:

        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False

    cars_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    cars_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=4,
                restore_best_weights=True),

            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=2,
                min_lr=1e-7)
        ]
    )

    # ========================================================
    # Save Stanford Cars model
    # ========================================================

    cars_model.save(STANFORD_MODEL_FILE)

    print("\n")
    print("Stanford Cars model saved:")
    print(STANFORD_MODEL_FILE)

    return cars_model


# ============================================================
# 7. Load target dataset
# ============================================================

def load_target_dataset():
    """
    Load 101_objectCategories.

    Expected:
        102 folders
        9,145 images
    """

    print("\n")
    print("########################################")
    print("# STEP 2: LOAD TARGET DATASET")
    print("########################################")

    if not os.path.isdir(TARGET_DIR):
        raise FileNotFoundError(
            f"Cannot find target dataset:\n"
            f"{TARGET_DIR}")

    train_ds = (
        keras.utils.image_dataset_from_directory(
            TARGET_DIR,
            validation_split=0.2,
            subset="training",
            seed=42,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="int"
        )
    )

    val_ds = (
        keras.utils.image_dataset_from_directory(
            TARGET_DIR,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="int"
        )
    )

    class_names = train_ds.class_names

    num_classes = len(class_names)

    print("\nTarget dataset classes:", num_classes)

    print("Target dataset class names:", class_names[0])

    print(class_names)

    # --------------------------------------------------------
    # Check expected class count
    # --------------------------------------------------------

    if num_classes != EXPECTED_TARGET_CLASSES:

        raise ValueError(
            "\nExpected "
            f"{EXPECTED_TARGET_CLASSES} "
            "target classes, but found "
            f"{num_classes}.\n\n"
            "Please check 101_objectCategories/."
        )

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return (train_ds, val_ds, class_names)


# ============================================================
# 8. Transfer Stanford Cars -> target dataset
# ============================================================

def train_transfer_model():
    """
    Transfer Stanford Cars learned features to
    101_objectCategories.

    Phase 1:
        Freeze Stanford Cars backbone.
        Train new 102-class head.

    Phase 2:
        Unfreeze top MobileNetV2 layers.
        Fine-tune on target dataset.

    Final:
        caltech101_model.h5
    """

    # ========================================================
    # STEP 1
    # Train Stanford Cars source model
    # ========================================================

    cars_model = train_stanford_cars()

    # --------------------------------------------------------
    # Extract trained Stanford Cars backbone
    # --------------------------------------------------------

    print("\n")
    print("########################################")
    print("# STEP 3: TRANSFER LEARNED FEATURES")
    print("########################################")

    base_model = None

    for layer in cars_model.layers:

        if (isinstance(layer,
                       keras.Model) and "mobilenetv2" in layer.name.lower()):
            base_model = layer
            break

    if base_model is None:

        raise RuntimeError(
            "Could not find MobileNetV2 backbone "
            "in Stanford Cars model.")

    print("Transferred backbone:", base_model.name)

    # ========================================================
    # STEP 2
    # Load target dataset
    # ========================================================

    (target_train_ds, target_val_ds, class_names) = load_target_dataset()

    NUM_CLASSES = len(class_names)

    # ========================================================
    # Build target model
    # ========================================================

    augmentation = build_data_augmentation()

    preprocess_input = (
        keras.applications.mobilenet_v2
        .preprocess_input)

    # --------------------------------------------------------
    # Freeze Stanford Cars backbone
    # --------------------------------------------------------

    base_model.trainable = False

    # --------------------------------------------------------
    # New target model
    # --------------------------------------------------------

    inputs = keras.Input(shape=IMG_SIZE + (3,), name="input")

    x = augmentation(inputs)

    x = preprocess_input(x)

    x = base_model(x, training=False)

    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(256, activation="relu")(x)

    x = keras.layers.Dropout(0.3)(x)

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Stanford Cars had a 196-class head.
    #
    # We REMOVE that classification head.
    #
    # This is a NEW 102-class target head.
    # --------------------------------------------------------

    outputs = keras.layers.Dense(
        NUM_CLASSES, activation="softmax",
        name="target_classifier")(x)

    model = keras.Model(inputs, outputs)

    print("\n")
    print("Target classifier output classes:", NUM_CLASSES)

    # ========================================================
    # TARGET PHASE 1
    # Frozen Stanford Cars backbone
    # ========================================================

    print("\n")
    print("########################################")
    print("# TARGET PHASE 1")
    print("# Frozen Stanford Cars backbone")
    print("########################################")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])

    model.fit(
        target_train_ds,
        validation_data=target_val_ds,
        epochs=15,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=4,
                restore_best_weights=True
            ),

            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=2,
                min_lr=1e-6
            )
        ]
    )

    # ========================================================
    # TARGET PHASE 2
    # Fine-tune Stanford Cars backbone
    # ========================================================

    print("\n")
    print("########################################")
    print("# TARGET PHASE 2")
    print("# Fine-tuning Stanford Cars features")
    print("########################################")

    # Unfreeze backbone.
    base_model.trainable = True

    # Freeze most layers.
    for layer in base_model.layers[:-40]:

        layer.trainable = False

    # Keep BatchNormalization frozen.
    for layer in base_model.layers:

        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False

    # --------------------------------------------------------
    # Compile again after changing trainable status.
    # --------------------------------------------------------

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # --------------------------------------------------------
    # Save best validation model
    # --------------------------------------------------------

    checkpoint = keras.callbacks.ModelCheckpoint(
        FINAL_MODEL_FILE,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    # --------------------------------------------------------
    # Fine-tune
    # --------------------------------------------------------

    history = model.fit(
        target_train_ds,
        validation_data=target_val_ds,
        epochs=20,
        callbacks=[
            checkpoint,

            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=2,
                min_lr=1e-7
            ),

            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=5,
                restore_best_weights=True
            )
        ]
    )

    # ========================================================
    # Final save
    # ========================================================

    model.save(
        FINAL_MODEL_FILE
    )

    # ========================================================
    # Report result
    # ========================================================

    best_val_accuracy = max(history.history["val_accuracy"])

    print("\n")
    print("########################################")
    print("# TRAINING COMPLETE")
    print("########################################")

    print(f"Best validation accuracy: "f"{best_val_accuracy:.4f}")

    print(f"Best validation accuracy (%): "f"{best_val_accuracy * 100:.2f}%")

    print("\nFinal model saved to:")

    print(FINAL_MODEL_FILE)

    if best_val_accuracy >= 0.85:

        print("\nSUCCESS: validation accuracy ""is >= 85%")

    else:

        print("\nWARNING: validation accuracy ""is below 85%.")

    return model


# ============================================================
# 9. Main
# ============================================================

if __name__ == "__main__":

    train_transfer_model()
