#!/usr/bin/env python3
"""computes and returns the feature importances"""

import numpy as np


def feature_importance(rf):
    """computes and returns the feature importance"""
    importances = rf.feature_importances_

    indices = np.argsort(importances)

    return importances, indices
