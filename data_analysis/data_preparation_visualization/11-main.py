#!/usr/bin/env python3

import pandas as pd
plot_numeric_vs_churn = __import__('11-plot_numeric_vs_churn').plot_numeric_vs_churn


df = pd.read_csv('precleaned-Telco-Customer-Churn.csv')
plot_numeric_vs_churn(df, 'tenure')