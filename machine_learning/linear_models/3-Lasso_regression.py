#!/usr/bin/env python3
"""creates and returns a Lasso Regression"""

from sklearn import linear_model


def lasso_regression(random_state):
    """creates and returns a Lasso regression"""
    model = linear_model.Lasso(random_state=random_state)
    return model
