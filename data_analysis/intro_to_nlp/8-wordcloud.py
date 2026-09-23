#!/usr/bin/env python3
"""generate word cloud from a preprocessed corpus"""
import wordcloud
import matplotlib.pyplot as plt


def generate_wordcloud(corpus_tokens, max_words=200, label=None):
    """Generates a word cloud from a list of tokens."""
    if not isinstance(corpus_tokens, list) or not corpus_tokens:
        return
    if not isinstance(max_words, int) or max_words < 1:
        raise ValueError("max_words must be a positive integer.")
    text = " ".join(str(t) for t in corpus_tokens)
    wc = wordcloud.WordCloud(max_words=max_words,
                             background_color="white",
                             width=800, height=400,
                             random_state=42).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    if label:
        plt.title(f"WordCloud — {label}")
    else:
        plt.title("WordCloud")
    plt.tight_layout()
    plt.savefig("8_task.png")
    plt.show()

    return wc
