#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
import math

Standardize = __import__('0-standardize').Standardize
Apply_PCA = __import__('1-pca').Apply_PCA
K_Means = __import__('2-k_means').K_Means
optimal_k = __import__('3-optimal_k').optimal_k

X, _ = load_wine(return_X_y=True)
X_scaled = Standardize(X)

ks, _, _ = optimal_k(X_scaled, max_clusters=10, random_state=2)

X_pca, pca_model = Apply_PCA(X_scaled, n_components=2, random_state=2)

cols = 3
rows = math.ceil(len(ks)/cols)
fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), squeeze=False)
sns.set(style="whitegrid")

for idx, k in enumerate(ks):
    row, col = divmod(idx, cols)
    ax = axes[row, col]

    model = K_Means(X_scaled, n_clusters=k, random_state=2)
    labels = model.labels_
    centroids_pca = pca_model.transform(model.cluster_centers_)

    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=60, alpha=0.7)

    centroid_plot = ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                               c='red', s=200, marker='X', label='Centroids')

    ax.set_title(f"K-means Clustering (k = {k})")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(handles=[centroid_plot])

for idx in range(len(ks), rows*cols):
    fig.delaxes(axes.flatten()[idx])

plt.tight_layout()
plt.show()
plt.savefig("Task_3.png")