#!/usr/bin/env python3
"""generate predictions from a trained tree-based classifier"""


def generate_predictions(clf, X):
    """generate predictions from a trained tree-basaed clssifier"""
    preds = clf.predict(X)
    return preds
