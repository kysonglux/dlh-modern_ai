#!/usr/bin/env python3
""" visualizes churn rates per category"""

import matplotlib.pyplot as plt


def plot_categorical_vs_churn(df, col):
    """visualizes churn rates per category """
    churn_rate = (
        df.groupby(col)["Churn"].apply(lambda x: (x == "Yes").mean())
    )

    plt.figure(figsize=(12, 8))
    churn_rate.plot(kind="bar")

    plt.title(f"Churn Rate by {col}")
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=45)

    plt.show()
    plt.savefig("Task_10.png")
