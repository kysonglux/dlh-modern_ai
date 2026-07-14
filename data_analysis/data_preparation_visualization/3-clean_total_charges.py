#!/usr/bin/env python3
""" data cleaning using Dropping, replacing, imputation """


def clean_total_charges(df, method='drop'):
    """data cleaning using Dropping, replacing, imputation """
    df = df.copy()

    if method == 'drop':
        df = df.dropna(subset=['TotalCharges'])
    elif method in ('median', 'impute'):
        value = (
            df['TotalCharges'].median()
            if method == 'median'
            else df['MonthlyCharges'] * df['tenure'])
        df['TotalCharges'] = df['TotalCharges'].fillna(value)
    return df
