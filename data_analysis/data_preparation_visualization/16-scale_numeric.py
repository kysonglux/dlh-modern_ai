#!/usr/bin/env python3
"""standardizes numeric columns"""

from sklearn import preprocessing


def scale_numeric(df):
    """ Standardizes numeric columns"""

    scaler = preprocessing.StandardScaler()

    cols = ["MonthlyCharges", "TotalCharges"]

    df[cols] = scaler.fit_transform(df[cols])

    return df
