#!/usr/bin/env python3
"""initialize a keras Tuner for hyperparameter tuning"""
import keras_tuner as kt


def initiate_tuner(tuner_type, build_model, seed,
                   hyperband_iterations, max_trials,
                   objective="val_accuracy"):
    """initialize a keras Tuner for hyperparameter tuning"""
    if tuner_type == 'Hyperband':
        tuner = kt.Hyperband(
            build_model,
            objective=objective,
            max_epochs=hyperband_iterations,
            factor=3,
            directory='my_dir',
            project_name='helloworld',
            overwrite=True
        )
    elif tuner_type == 'RandomSearch':
        tuner = kt.RandomSearch(
            build_model,
            objective=objective,
            max_trials=max_trials,
            directory='my_dir',
            project_name='helloworld',
            overwrite=True
        )
    elif tuner_type == 'BayesianOptimization':
        tuner = kt.BayesianOptimization(
            build_model,
            objective=objective,
            max_trials=max_trials,
            directory='my_dir',
            project_name='helloworld',
            overwrite=True
        )
    else:
        raise ValueError(f"Unknown tuner type: {tuner_type}")
    return tuner
