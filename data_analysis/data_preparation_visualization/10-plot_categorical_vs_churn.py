#!/usr/bin/env python3
""" visualizes churn rates per category"""

import matplotlib.pyplot as plt


def plot_categorical_vs_churn(df, col):
    """visualizes churn rates per category """
    churn_rate = (
        df.groupby(col)["Churn"].apply(lambda x: (x == "Yes").mean())
    )

    plt.figure(figsize=(12, 8))
    plt.bar(
        churn_rate.index,
        churn_rate.values
    )

    plt.title(f"Churn Rate by {col}")
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=45)

    plt.show()
    plt.savefig("Task_10.png")
