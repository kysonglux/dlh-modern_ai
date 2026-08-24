#!/usr/bin/env python3
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import numpy as np
import random
import os
create_cnn_model = __import__('0-create_cnn_model').create_cnn_model

SEED = 0
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
_, (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

test_images = test_images.astype('float32') / 255.0
test_images = test_images.reshape(-1, 28, 28, 1)

model = create_cnn_model(
    input_shape=(28, 28, 1),
    filters=[32, 64],
    kernel_sizes=[(3, 3), (3, 3)],
    activations=['relu', 'relu'],
    pooling_type='max'
)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2%}")
