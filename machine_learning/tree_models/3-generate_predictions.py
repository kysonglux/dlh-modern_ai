#!/usr/bin/env python3
"""generate predictions from a trained tree-based classifier"""

import numpy as np
import pandas as pd


def generate_predictions(clf, X):
    """generate predictions from a trained tree-basaed clssifier"""
    if not hasattr(clf, "predict"):
        raise TypeError()
    if not isinstance(X, (np.ndarray, pd.DataFrame)):
        raise TypeError()
    preds = clf.predict(X)
    return np.asarray(preds)
