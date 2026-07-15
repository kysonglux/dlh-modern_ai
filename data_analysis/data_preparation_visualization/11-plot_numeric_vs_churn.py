#!/usr/bin/env python3
""" compares continuous numeric feature distributions by churn"""

import matplotlib.pyplot as plt


def plot_numeric_vs_churn(df, col):
    """compares continuous numeric feature distribution by churn"""
    churn_yes = df[df["Churn"] == "Yes"][col]
    churn_no = df[df["Churn"] == "No"][col]

    plt.figure(figsize=(12, 8))

    plt.hist(
        [churn_no, churn_yes],
        bins=30,
        label=["No", "Yes"]
    )

    plt.title(f"{col} Distribution by Churn")
    plt.xlabel(col)
    plt.legend(title="Churn")

    plt.show()
    plt.savefig("Task_11.png")
