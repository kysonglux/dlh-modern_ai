#!/usr/bin/env python3
"""Welch's tests for continuous numeric features using scipy"""

from scipy import stats


def ttest_numeric(df):
    """Welch:s tests for continuous numeric features using scipy"""
    results = {}
    numeric_cols = df.select_dtypes(include=["number"]).columns

    churn_yes = df[df["Churn"] == "Yes"]
    churn_no = df[df["Churn"] == "No"]

    for col in numeric_cols:
        yes_vals = churn_yes[col].dropna()
        no_vals = churn_no[col].dropna()

        _, p_value = stats.ttest_ind(yes_vals, no_vals, equal_var=False)

        results[col] = p_value
    return results
