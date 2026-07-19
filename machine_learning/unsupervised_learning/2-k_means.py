#!/usr/bin/env python3
"""creates and fits a K-Means clustering model"""

from sklearn import cluster


def K_Means(X, n_clusters, random_state):
    """creates and fits a K-Means clustering model"""
    new = cluster.KMeans(n_clusters=n_clusters, random_state=random_state)
    return new.fit(X)
