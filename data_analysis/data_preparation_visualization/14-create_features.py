#!/usr/bin/env python3
""" engineers new features from the dataset"""

import pandas as pd


def create_features(df):
    """engineers new features from the dataset"""
    service_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    service_df = df[service_cols].copy()

    service_df["InternetService"] = service_df["InternetService"].apply(
        lambda x: "Yes" if x in ["DSL", "Fiber optic"] else "No"
    )

    df["NumServices"] = (service_df == "Yes").sum(axis=1)

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 60, float("inf")],
        labels=["0-12", "13-24", "25-48", "49-60", "60+"],
        right=True,
        include_lowest=False
    )

    df.drop(columns=service_cols, inplace=True)
    df.drop(columns=["tenure"], inplace=True)
    return df
