#!/usr/bin/env python3

import pandas as pd
convert_columns = __import__('2-convert_columns').convert_columns
plot_missingness = __import__('1-plot_missingness').plot_missingness

df = pd.read_csv('Telco-Customer-Churn.csv')
df = convert_columns(df)
print("SeniorCitizen column type after conversion:", df['SeniorCitizen'].dtype)
print("Unique values in the 'SeniorCitizen' column:", df['SeniorCitizen'].unique())
print("\nTotalCharges column type after conversion:", df['TotalCharges'].dtype)
print("Missing values in TotalCharges after conversion:", df['TotalCharges'].isna().sum())
plot_missingness(df)