#!/usr/bin/env python3

import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

Standardize = __import__('0-standardize').Standardize
K_Means = __import__('2-k_means').K_Means
Apply_PCA = __import__('1-pca').Apply_PCA


X, _ = load_wine(return_X_y=True)

model = K_Means(X, n_clusters=3, random_state=2)
labels = model.labels_

X_orig_2d, _ = Apply_PCA(X, n_components=2, random_state=2)

X_scaled = Standardize(X)
model_scaled = K_Means(X_scaled, n_clusters=3, random_state=2)
labels_scaled = model_scaled.labels_

X_scaled_2d, _ = Apply_PCA(X_scaled, n_components=2, random_state=2)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original data
scatter0 = axes[0].scatter(X_orig_2d[:, 0], X_orig_2d[:, 1], c=labels, cmap='viridis', s=50)
axes[0].set_title('K-Means Clustering in 2D PCA Space (Original Data)')
axes[0].set_xlabel('PCA Component 1')
axes[0].set_ylabel('PCA Component 2')
legend0 = axes[0].legend(*scatter0.legend_elements(), title="Clusters")
axes[0].add_artist(legend0)

# Standardized data
scatter1 = axes[1].scatter(X_scaled_2d[:, 0], X_scaled_2d[:, 1], c=labels_scaled, cmap='viridis', s=50)
axes[1].set_title('K-Means Clustering in 2D PCA Space (Standardized Data)')
axes[1].set_xlabel('PCA Component 1')
axes[1].set_ylabel('PCA Component 2')
legend1 = axes[1].legend(*scatter1.legend_elements(), title="Clusters")
axes[1].add_artist(legend1)

plt.tight_layout()
plt.show()
plt.savefig("Task_2.png")