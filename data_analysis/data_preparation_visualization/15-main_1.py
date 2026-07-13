#!/usr/bin/env python3

import pandas as pd
create_features = __import__('14-create_features').create_features
encode_features = __import__('15-encode_features').encode_features


df = pd.read_csv('precleaned-Telco-Customer-Churn.csv')
df.drop(columns=['gender', 'PhoneService'], inplace=True)
df = create_features(df)
df_enc, churn_le, binary_oe, tenure_oe = encode_features(df)

print(df_enc.dtypes)