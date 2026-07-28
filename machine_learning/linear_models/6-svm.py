#!/usr/bin/env python3
"""create a Support vector machine(SVM) classifier"""

from sklearn import svm


def get_SVM_model(name, random_state):
    """create a Support vector machine(SVM) classifier"""
    if name == 'linear':
        model = svm.SVC(kernel='linear', random_state=random_state)

    elif name == 'poly':
        model = svm.SVC(kernel='poly', random_state=random_state)

    elif name == 'rbf':
        model = svm.SVC(kernel='rbf', random_state=random_state)

    else:
        raise ValueError(f"Unknown model name '{name}'")

    return model
