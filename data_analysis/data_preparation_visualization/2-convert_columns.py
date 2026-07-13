#!/usr/bin/env python3
"""Changing Column Types """

import pandas as pd


def convert_columns(df):
    """changing Column Types"""
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    return df
