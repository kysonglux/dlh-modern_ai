#!/usr/bin/env python3
""" splits data into tran/test sets"""
from sklearn import model_selection


def split_data(df, target='Churn', test_size=0.2, random_state=42):
    """Splits the DataFrame into train/test sets"""

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
