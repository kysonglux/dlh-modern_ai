#!/usr/bin/env python3

from sklearn.datasets import load_wine
Standardize = __import__('0-standardize').Standardize
Apply_PCA = __import__('1-pca').Apply_PCA


X, _ = load_wine(return_X_y=True)

X_scaled = Standardize(X)

X_pca, pca = Apply_PCA(X_scaled, n_components=None, random_state=2)


print("Original shape:", X_scaled.shape)
print("PCA-transformed shape:", X_pca.shape)
print("\nExplained variance ratio of components:", pca.explained_variance_ratio_)