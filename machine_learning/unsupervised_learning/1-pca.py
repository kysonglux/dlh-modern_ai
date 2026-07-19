#!/usr/bin/env python3
"""PCA: dimentionality reduction"""

from sklearn import decomposition


def Apply_PCA(X, n_components, random_state):
    """dimetionality reduction using PCA"""
    new = decomposition.PCA(n_components=n_components,
                            random_state=random_state)
    result = new.fit_transform(X)
    return result, new
