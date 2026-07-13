#!/usr/bin/env python3

import pandas as pd
plot_correlation_heatmap = __import__('9-plot_correlation_heatmap').plot_correlation_heatmap


df = pd.read_csv('precleaned-Telco-Customer-Churn.csv')
plot_correlation_heatmap(df)
