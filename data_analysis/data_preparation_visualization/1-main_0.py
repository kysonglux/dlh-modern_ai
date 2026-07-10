#!/usr/bin/env python3

import pandas as pd
plot_missingness = __import__('1-plot_missingness').plot_missingness


df = pd.read_csv('Telco-Customer-Churn.csv')
plot_missingness(df)
