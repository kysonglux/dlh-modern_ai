#!/usr/bin/env python3
"""perform a grid search for the best prepruning"""

from sklearn import model_selection


def prepruning(X, y, clf):
    """perform a grid search for the best prepruning"""
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [2, 3, 4],
        "min_samples_leaf": [2, 3, 4],
        "min_samples_split": [2, 3, 4]
    }

    grid = model_selection.GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=5
    )

    grid.fit(X, y)

    return grid.best_params_
