#!/usr/bin/env python3
"""Agglomerative hierarchical clustering"""

from sklearn import cluster
from sklearn import metrics
Apply_PCA = __import__('1-pca').Apply_PCA


def Agglomerative_Clustering(X, n_clusters,
                             random_state, n_components, use_pca_data=True):
    """Agglomerative hierarchical clustering"""
    if use_pca_data is True:
        data_used, PCA_instance = Apply_PCA(X, n_components, random_state)
    else:
        data_used = X

    model = cluster.AgglomerativeClustering(n_clusters=n_clusters,
                                            linkage='ward')
    model.fit(data_used)

    if n_clusters > 1:
        score = metrics.silhouette_score(data_used, model.labels_)
    else:
        score = None

    return model, data_used, score
