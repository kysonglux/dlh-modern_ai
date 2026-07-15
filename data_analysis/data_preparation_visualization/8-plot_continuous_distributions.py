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
        df = df[[col for col in columns_to_plot
                if col in df.colunmns
                and np.issubdtype(df[col].dtype, np.number)]
                ]

    n_cols = df.shape[1]
    fig, axes = plt.subplots(n_cols, 2, figsize=(10, 3*n_cols))

    if n_cols == 1:
        axes = axes.reshape(1, -1)

    for i, col in enumerate(df.columns):
        data = df[col].dropna().values

        ax_hist = axes[i][0]
        ax_hist.hist(
            data,
            bins=30,
            density=True,
            alpha=0.7,
            edgecolor='black'
        )

        kde = stats.gaussian_kde(data)
        x_vals = np.linspace(data.min(), data.max(), 500)
        ax_hist.plot(x_vals, kde(x_vals), color='red', linestyle='--')

        ax_hist.set_title(f"{col} Histogram + KDE")

        if col == "TotalCharges":
            max_density = kde(x_vals).max()
            yticks = np.arange(0, max_density + 0.0004, 0.0002)
            ax_hist.set_yticks(yticks)

        ax_box = axes[i][1]
        ax_box.boxplot(data, vert=False)
        ax_box.set_title(f"{col} Boxplot")

    plt.tight_layout()
    plt.savefig("Task_8.png")
    plt.show()
