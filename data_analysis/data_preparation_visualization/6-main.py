#!/usr/bin/env python3

import pandas as pd
plot_churn_distribution = __import__('6-plot_churn_distribution').plot_churn_distribution


df = pd.read_csv('precleaned-Telco-Customer-Churn.csv')
plot_churn_distribution(df)