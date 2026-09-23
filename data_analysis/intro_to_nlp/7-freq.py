#!/usr/bin/env python3
"""most frequent tokens in a preprocessesd corpus"""
import nltk
import matplotlib.pyplot as plt


def plot_top_n_frequencies(corpus_tokens, n=20):
    """Plots the top n most frequent tokens in a corpus."""
    if not isinstance(corpus_tokens, list) or not corpus_tokens:
        return
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer.")

    corpus_tokens = [str(t) for t in corpus_tokens]
    freq_dist = nltk.FreqDist(corpus_tokens)
    top_n = freq_dist.most_common(n)
    tokens, frequencies = zip(*top_n)
    plt.figure(figsize=(12, 5))
    plt.tight_layout()
    plt.bar(tokens, frequencies)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.title(f"Top {n} Most Frequent Words")
    plt.savefig("7_task.png")
    plt.show()
    return freq_dist
