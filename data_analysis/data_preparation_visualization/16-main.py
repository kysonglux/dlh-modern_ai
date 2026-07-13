#!/usr/bin/env python3

import pandas as pd
create_features = __import__('14-create_features').create_features
encode_features = __import__('15-encode_features').encode_features
scale_numeric = __import__('16-scale_numeric').scale_numeric


df = pd.read_csv('precleaned-Telco-Customer-Churn.csv')
df.drop(columns=['gender', 'PhoneService'], inplace=True)
df = create_features(df)
df_enc, _, _, _ = encode_features(df)
df_scaled = scale_numeric(df_enc)
print(df_scaled[['MonthlyCharges','TotalCharges']].describe().loc[['mean','std']])