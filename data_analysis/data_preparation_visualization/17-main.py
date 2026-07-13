#!/usr/bin/env python3

import pandas as pd
create_features = __import__('14-create_features').create_features
encode_features = __import__('15-encode_features').encode_features
scale_numeric = __import__('16-scale_numeric').scale_numeric
split_data = __import__('17-split_data').split_data


df = pd.read_csv('precleaned-Telco-Customer-Churn.csv')
df.drop(columns=['gender', 'PhoneService'], inplace=True)
df = create_features(df)
df_enc, _, _, _ = encode_features(df)
df_scaled = scale_numeric(df_enc)
X_train, X_test, y_train, y_test = split_data(df_scaled)
print("Train:", X_train.shape, "Test:", X_test.shape)
print("Train Churn %:", y_train.mean(), "Test Churn %:", y_test.mean())
