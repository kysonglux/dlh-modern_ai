#!/usr/bin/env python3
"""create a logistic regression model, performs binary classification"""

from sklearn import linear_model


def Logistic_Regression_Model(random_state):
    """create a logistic regression model, performs binary classification"""
    model = linear_model.LogisticRegression(random_state=random_state)
    return model
