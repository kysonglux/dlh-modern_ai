#!/usr/bin/env python3
"""standardizes tabular data using Scikit-learn"""

from sklearn import preprocessing


def Standardize(X):
    """standardizes tabular data using Scikit-learn"""
    scaler = preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(X)
    return x_scaled
