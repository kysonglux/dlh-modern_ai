#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
import numpy as np

Standardize = __import__('0-standardize').Standardize
Agglomerative_Clustering = __import__('4-agglomerative').Agglomerative_Clustering


X, _ = load_wine(return_X_y=True)
X_scaled = Standardize(X)

ks = [2, 3, 4]

cols = 2
rows = len(ks)
fig, axes = plt.subplots(rows, cols, figsize=(12, 5*rows), squeeze=False)
sns.set(style="whitegrid")

for row_idx, k in enumerate(ks):
    for col_idx, use_pca in enumerate([True, False]):
        ax = axes[row_idx, col_idx]

        model, X_used, score = Agglomerative_Clustering(
            X_scaled, n_clusters=k, random_state=2, n_components=5, use_pca_data=use_pca
        )

        labels = model.labels_

        ax.scatter(X_used[:, 0], X_used[:, 1], c=labels, cmap='tab10', s=60, alpha=0.7)

        centroids = np.array([X_used[labels == i].mean(axis=0) for i in range(k)])
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroids')

        data_type = "PCA-reduced" if use_pca else "Original"
        score_text = f"Silhouette Score: {score:.2f}" if score is not None else ""
        ax.set_title(f"k = {k} ({data_type}) | {score_text}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend()

plt.tight_layout()
plt.show()
plt.savefig("Task_4.png")