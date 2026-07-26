#!/usr/bin/env python3
"""create a random forest classifier """

from sklearn import ensemble


def random_forest(n_estimators, random_state):
    """create a rondom forest classifier"""
    model = ensemble.RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
    return model
