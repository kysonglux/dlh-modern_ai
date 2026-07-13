#!/usr/bin/env python3

import pandas as pd
plot_categorical_vs_churn = __import__('10-plot_categorical_vs_churn').plot_categorical_vs_churn


df = pd.read_csv('precleaned-Telco-Customer-Churn.csv')
plot_categorical_vs_churn(df, 'PhoneService')
