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
    bi_oe = preprocessing.OrdinalEncoder(categories=[["No", "Yes"]])

    for col in bi_cols:
        df[col] = bi_oe.fit_transform(df[[col]]).astype(int)

    '#3. Ordinal encode TenureGroup(alphabetical order)'

    tenure_oe = preprocessing.OrdinalEncoder()
    sorted_tenure = sorted(df["TenureGroup"].unique())
    tenure_oe.fit([[cat] for cat in sorted_tenure])
    df["TenureGroup"] = tenure_oe.transform(df[["TenureGroup"]]).astype(int)

    '#4.One-hot encode Contract and paymentMethod (drop first)'
    df = pd.get_dummies(
        df,
        columns=["Contract", "PaymentMethod"],
        drop_first=True)

    return df, churn_le, bi_oe, tenure_oe
