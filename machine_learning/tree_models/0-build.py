#!/usr/bin/env python3
"""create a decision tree classifier using scikit-learn"""

from sklearn import tree


def build_decision_tree(min_samples_leaf, min_samples_split, random_state):
    """create a decision tree classifier using scikit-learn"""
    model = tree.DecisionTreeClassifier(
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=random_state)
    return model
