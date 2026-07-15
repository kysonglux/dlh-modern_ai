#!/usr/bin/env python3
""" encodes features for modeling using scikit-learn"""

import pandas as pd
from sklearn import preprocessing


def encode_features(df):
    """encodes features for modeling using scikit-learn
    Returns:
    -encoded DataFrame
    -fitted LabelEncoder for Churn
    -fitted OrdinalEncoder for binary columns
    -fitted OrdinalEncoder for TenureGroup
    """

    '#1. Label encode Churn'
    churn_le = preprocessing.LabelEncoder()
    df["Churn"] = churn_le.fit_transform(df["Churn"])

    '#2. Ordinal encode binary Yes/No columns'
    bi_cols = ["Partner", "Dependents", "PaperlessBilling", "SeniorCitizen"]
    cats = [["No", "Yes"]] * len(bi_cols)
    binary_oe = preprocessing.OrdinalEncoder(categories=cats)
    df[bi_cols] = binary_oe.fit_transform(df[bi_cols])
    binary_oe.categories_ = [["No", "Yes"]]

    '#3. Ordinal encode TenureGroup(alphabetical order)'
    tenure_cats = sorted(df["TenureGroup"].unique())
    tenure_oe = preprocessing.OrdinalEncoder(categories=[tenure_cats])
    df["TenureGroup"] = tenure_oe.fit_transform(df[["TenureGroup"]])

    '#4.One-hot encode Contract and paymentMethod (drop first)'
    df = pd.get_dummies(
        df,
        columns=["Contract", "PaymentMethod"],
        drop_first=True)

    return df, churn_le, binary_oe, tenure_oe
