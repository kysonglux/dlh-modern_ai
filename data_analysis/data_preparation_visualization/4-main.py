#!/usr/bin/env python3

import pandas as pd
convert_columns = __import__('2-convert_columns').convert_columns
clean_total_charges = __import__('3-clean_total_charges').clean_total_charges
remove_duplicates = __import__('4-remove_duplicates').remove_duplicates


df = pd.read_csv('Telco-Customer-Churn.csv')
df = convert_columns(df)
df = clean_total_charges(df)
dupl_before = df.duplicated().sum()
df = remove_duplicates(df)
print("Duplicates before:", dupl_before)
print("Duplicates after:", df.duplicated().sum())