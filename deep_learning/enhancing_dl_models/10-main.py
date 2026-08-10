#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import os
import random

build_model = __import__('8-build_model_to_be_tuned').build_model
initiate_tuner = __import__('9-initiate_tuner').initiate_tuner
search_and_return_best_model = __import__('10-search').search_and_return_best_model


def set_seed(SEED):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

set_seed(0)

seed = 0
hyperband_iterations = 5
max_trials = 5
objective = 'val_accuracy'
epochs = 10
validation_split = 0.2

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, 10)

results = []

for tuner_name in ['Hyperband', 'RandomSearch', 'BayesianOptimization']:

    tuner = initiate_tuner(tuner_name, build_model, seed, hyperband_iterations, max_trials, objective)

    best_hyperparameters = search_and_return_best_model(
        tuner, x_train, y_train, epochs, validation_split, verbose=1
    )

    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]

    results.append({
        'Tuner': tuner_name,
        'Best Accuracy': best_trial.score,
        'Best Hyperparameters': best_hyperparameters.values
    })

print(results)
