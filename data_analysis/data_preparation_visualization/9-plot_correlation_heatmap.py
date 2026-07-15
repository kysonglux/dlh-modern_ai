#!/usr/bin/env python3
"""visualizes correlations between continuous numeric features"""

import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation_heatmap(df):
    """visualizes correlations between continuous numeric features"""
    plt.figure(figsize=(6, 5))

    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()
    plt.savefig("Task_9.png")
