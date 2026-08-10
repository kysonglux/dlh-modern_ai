#!/usr/bin/env python3

import tensorflow as tf
import os
import random

build_model = __import__('8-build_model_to_be_tuned').build_model
initiate_tuner = __import__('9-initiate_tuner').initiate_tuner


def set_seed(SEED):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

set_seed(0)

seed = 0
hyperband_iterations = 5
max_trials = 5
objective = 'val_accuracy'

for tuner_type in ["Hyperband", "RandomSearch", "BayesianOptimization"]:
    print(f"\n== {tuner_type} Tuner ==")
    tuner = initiate_tuner(tuner_type, build_model, seed, hyperband_iterations, max_trials, objective)

    print(f"Tuner Type: {tuner_type}")
    print(f"Tuner Objective: {tuner.oracle.objective.name}")
    if tuner_type == "RandomSearch" or tuner_type == "BayesianOptimization":
      print(f"Max Trials: {tuner.oracle.max_trials}")
    elif tuner_type == "Hyperband":
        print(f"Hyperband Iterations: {hyperband_iterations}")
    print("\n")
    tuner.search_space_summary()