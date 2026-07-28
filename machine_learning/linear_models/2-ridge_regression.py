#!/usr/bin/env python3
"""creates and returns a ridge Regression"""

from sklearn import linear_model


def ridge_regression(random_state):
    """creates and returns a rigde Regression"""
    model = linear_model.Ridge(random_state=random_state)
    return model
