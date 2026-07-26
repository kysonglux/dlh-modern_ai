#!/usr/bin/env python3
"""initializes and returns an untrained boosting classifier """

from sklearn import ensemble
import xgboost as xgb
import lightgbm as lgb


def compare_boosting_classifiers(name, n_estimators, random_state):
    """initializes and returns an untrined boosting classifier"""
    if name == "adaboost":
        return ensemble.AdaBoostClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )

    elif name == "gradientboosting":
        return ensemble.GradientBoostingClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )

    elif name == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )

    elif name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            verbose=-1
        )

    raise ValueError(f"Unknown model name '{name}'")
