#!/usr/bin/env python3
"""train a tree based classifier using scikit-learn"""


def train_tree(clf, X, y):
    """train a tree based classifier using scikit-learn"""
    clf.fit(X, y)
