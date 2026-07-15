#!/usr/bin/env python3

import pandas as pd
create_features = __import__('14-create_features').create_features


df = pd.read_csv('precleaned-Telco-Customer-Churn.csv')
df.drop(columns=['gender', 'PhoneService'], inplace=True)
df = create_features(df)
print(df[['NumServices','TenureGroup']].head())