#!/usr/bin/env python3
""" drop columns """


def drop_customerID(df):
    """drop columns """
    df = df.drop(columns=['customerID'])
    return df
