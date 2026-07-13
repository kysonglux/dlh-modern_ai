#!/usr/bin/env python3

import pandas as pd
convert_columns = __import__('2-convert_columns').convert_columns
clean_total_charges = __import__('3-clean_total_charges').clean_total_charges


df = pd.read_csv('Telco-Customer-Churn.csv')
df = convert_columns(df)
df1 = clean_total_charges(df)
df2 = clean_total_charges(df, method='impute')
df3 = clean_total_charges(df, method='median')
nan_indices = df[df['TotalCharges'].isna()].index
print("Missing after drop:", df1['TotalCharges'].isna().sum())
print("Missing after impute:", df2.loc[nan_indices, 'TotalCharges'])
print("Missing after median:", df3.loc[nan_indices, 'TotalCharges'])