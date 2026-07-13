#!/usr/bin/env python3

import pandas as pd
chi_square_tests = __import__('12-chi_square_tests').chi_square_tests


df = pd.read_csv('precleaned-Telco-Customer-Churn.csv')
print(chi_square_tests(df))