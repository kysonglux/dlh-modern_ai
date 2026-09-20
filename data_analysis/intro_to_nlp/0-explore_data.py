#!/usr/bin/env python3
""" dataset exploration """
import matplotlib.pyplot as plt
import seaborn as sns


def explore_data(df):
    """initial dataset exploration"""
    df["msg_length"] = df["message"].apply(len)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    label_counts = df['label'].value_counts().reset_index()
    label_counts.columns = ['label', 'count']
    axes[0].set_title("Ham vs Spam Counts")
    sns.barplot(x='label', y='count', data=label_counts, ax=axes[0])

    axes[1].set_title("Histogram of Raw Message Lengths")
    sns.histplot(x="msg_length", bins=50, data=df, ax=axes[1])
    axes[1].set_xlabel("length")
    axes[1].set_ylabel("count")
    plt.tight_layout()
    plt.show()
    plt.savefig("0_task")
