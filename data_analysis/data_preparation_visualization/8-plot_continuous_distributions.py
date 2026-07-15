#!/usr/bin/env python3
""" visualizes the distributions of continuous numerical features"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_continuous_distributions(df, columns_to_plot=None):
    """visualizes the distributions of continuous numerical features"""

    if columns_to_plot is None:
        df = df.select_dtypes(include=[np.number])
    else:
        df = df[columns_to_plot]

    n_cols = df.shape[1]
    fig, axes = plt.subplots(n_cols, 2, figsize=(10, 3*n_cols))

    if n_cols == 1:
        axes = axes.reshape(1, -1)

    for i, col in enumerate(df.columns):
        data = df[col].dropna().values

        axes[i, 0].hist(
            data,
            bins=30,
            density=True,
            alpha=0.7,
            edgecolor='black'
        )

        kde = stats.gaussian_kde(data)
        x_vals = np.linspace(data.min(), data.max(), 200)
        axes[i, 0].plot(
            x_vals,
            kde(x_vals),
            color='red',
            linestyle='--'
            )

        axes[i, 0].set_title(f"{col} Histogram + KDE")

        axes[i, 1].boxplot(data, vert=False)
        axes[i, 1].set_title(f"{col} Boxplot")

    plt.tight_layout()
    plt.savefig("Task_8.png")
    plt.show()
