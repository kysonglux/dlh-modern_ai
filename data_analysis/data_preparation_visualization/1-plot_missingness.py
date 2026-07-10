#!/usr/bin/env python3
"""Visualize missing values in a DataFrame"""

import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """Visualize missing values in a DataFrame"""
    
    plt.figure(figsize=(12, 8))
    missing = df.isna()
    rows_missing, cols_missing = np.where(missing)

    column_name = df.columns.tolist()

    y_value = [idx for idx in cols_missing]
    y_tick_labels = column_name
    y_tick_positions = np.arrange(len(column_name))

    plt.scatter(rows_missing, y_value, marker='|')
    plt.yticks(y_tick_positions, y_tick_labels)

    plt.title("Missingness Plot")
    plt.ylable("Column_name")
    plt.tight_layout()
    plt.show()