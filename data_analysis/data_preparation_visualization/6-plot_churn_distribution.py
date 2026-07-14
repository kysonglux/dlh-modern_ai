#!/usr/bin/env python3
""" plot the bar chart """

import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """plot the bar chart """
    plt.figure(figsize=(12, 8))
    churn_counts = df['Churn'].value_counts()
    colors = {'No': 'skyblue', 'Yes': 'salmon'}

    categories = churn_counts.index.tolist()
    counts = churn_counts.values.tolist()

    plot_colors = [colors[index] for index in categories]

    plt.bar(categories, counts, color=plot_colors)
    plt.title("Churn Distribution")
    plt.ylabel('Count')
    plt.savefig('6.bar_chart.png')
