#!/usr/bin/env python3

import pandas as pd
convert_columns = __import__('2-convert_columns').convert_columns
clean_total_charges = __import__('3-clean_total_charges').clean_total_charges
remove_duplicates = __import__('4-remove_duplicates').remove_duplicates
drop_customerID = __import__('5-drop_customerID').drop_customerID


df = pd.read_csv('Telco-Customer-Churn.csv')
df = convert_columns(df)
df = clean_total_charges(df)
df = remove_duplicates(df)
df = drop_customerID(df)
print("Columns after removing customerID:", df.columns.tolist())

df.to_csv('precleaned-Telco-Customer-Churn.csv', index=False)