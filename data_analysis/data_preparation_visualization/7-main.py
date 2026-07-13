#!/usr/bin/env python3

import pandas as pd
plot_categorical_distributions = __import__('7-plot_categorical_distributions').plot_categorical_distributions


df = pd.read_csv('precleaned-Telco-Customer-Churn.csv')
plot_categorical_distributions(df)