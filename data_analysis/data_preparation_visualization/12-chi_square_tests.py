#!/usr/bin/env python3
""" chi-square tests for categorical features """

import pandas as pd
from scipy import stats


def chi_square_tests(df):
    """chi-square tests for categorical features"""
    results = {}

    categorical_cols = df.select_dtypes(include=["object"]).columns
    categorical_cols = [col for col in categorical_cols if col != "Churn"]

    for col in categorical_cols:
        contigency_table = pd.crosstab(df[col], df["Churn"])
        chi2, p_value, _, _ = stats.chi2_contingency(contigency_table)
        results[col] = p_value
    return results
