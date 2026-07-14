#!/usr/bin/env python3
""" drop duplicate """


def remove_duplicates(df):
    """drop duplicate """
    df = df.drop_duplicates()
    return df
