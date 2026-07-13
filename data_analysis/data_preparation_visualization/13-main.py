#!/usr/bin/env python3

import pandas as pd
ttest_numeric = __import__('13-ttest_numeric').ttest_numeric


df = pd.read_csv('precleaned-Telco-Customer-Churn.csv')
print(ttest_numeric(df))