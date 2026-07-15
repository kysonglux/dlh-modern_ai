#!/usr/bin/env python3
""" visualizes categorical feature distributions """

import matplotlib.pyplot as plt


def plot_categorical_distributions(df, columns_to_plot=None):
    """ visualizes categorical feature distributions """
    df = df.copy()

    if columns_to_plot is None:
        df = df.drop(columns=['Churn'], errors='ignore')
        df = df.select_dtypes(include=['object'])
    else:
        df = df[columns_to_plot]

    total_columns = df.shape[1]
    n_cols, n_rows = (3, (total_columns + 3 - 1) // 3)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))

    axes = axes.flatten()

    for i, col in enumerate(df.columns):
        counts = df[col].value_counts()
        x = counts.index.tolist()
        y = counts.values

        axes[i].bar(x, y)
        axes[i].set_title(col)

        axes[i].tick_params(axis="x", rotation=45)

    for j in range(total_columns, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig("Task_7.png")
    plt.show()
