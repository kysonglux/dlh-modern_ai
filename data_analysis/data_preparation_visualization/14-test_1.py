#!/usr/bin/env python3

import pandas as pd
plot_categorical_vs_churn = __import__('10-plot_categorical_vs_churn').plot_categorical_vs_churn
chi_square_tests = __import__('12-chi_square_tests').chi_square_tests
create_features = __import__('14-create_features').create_features


df = pd.read_csv('precleaned-Telco-Customer-Churn.csv')
df.drop(columns=['gender', 'PhoneService'], inplace=True)
df = create_features(df)
df['TenureGroup'] = df['TenureGroup'].astype('object')

results = chi_square_tests(df)
print(results['TenureGroup'])
plot_categorical_vs_churn(df, 'TenureGroup')
