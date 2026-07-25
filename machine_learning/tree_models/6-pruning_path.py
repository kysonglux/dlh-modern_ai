#!/usr/bin/env python3
"""retrieves the cost-complexity pruning path"""


def get_pruning_path(clf, X, y):
    """retrieves the costs-complexity pruning path"""
    clf.fit(X, y)

    path = clf.cost_complexity_pruning_path(X, y)

    ccp_alphas = path.ccp_alphas
    impurities = path.impurities

    return ccp_alphas, impurities
