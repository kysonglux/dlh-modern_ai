#!/usr/bin/env python3

import pandas as pd
plot_continuous_distributions = __import__('8-plot_continuous_distributions').plot_continuous_distributions


df = pd.read_csv('precleaned-Telco-Customer-Churn.csv')
plot_continuous_distributions(df)