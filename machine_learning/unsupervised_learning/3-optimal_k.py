#!/usr/bin/env python3
"""evaluates K-Means clustering quality using silhouette scores"""

from sklearn import metrics
K_Means = __import__('2-k_means').K_Means


def optimal_k(X, max_clusters, random_state):
    """dvaluates K-Means clustering quality using silhouette scores"""
    cluster_numbers = []
    inerias = []
    scores = []

    for i in range(2, max_clusters + 1):
        cluster_numbers.append(i)
        model = K_Means(X, i, random_state)
        inerias.append(model.inertia_)
        scores.append(metrics.silhouette_score(X, model.labels_))
    return cluster_numbers, inerias, scores
