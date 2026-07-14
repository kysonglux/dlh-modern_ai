#!/usr/bin/env python3
""" visualizes categorical feature distributions """

import matplotlib.pyplot as plt


def plot_categorical_distributions(df, columns_to_plot=None):
    """ visualizes categorical feature distributions """
    if columns_to_plot is None:
        columns_to_plot = [
            col for col in df.columns
            if df[col].dtype == 'object' and col != "Churn"
        ]
    else:
        columns_to_plot = [
            col for col in columns_to_plot
            if col in df.columns and df[col].dtype == 'object'
        ]
    n_cols = 2
    n_rows = (len(columns_to_plot) + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))

    axes = axes.flatten()

    for ax, col in zip(axes, columns_to_plot):
        value_counts = df[col].value_counts()

        ax.bar(value_counts.index, value_counts.values)
        ax.set_title(col)

        ax.set_xticks(range(len(value_counts.index)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')

    for i in range(len(columns_to_plot), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("Task_7.png")
    plt.show()
