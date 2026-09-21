#!/usr/bin/env python3
"""remove_stopwords.py — Remove stopwords from a list of tokens."""
import nltk
nltk.download('stopwords')


def remove_stopwords(tokens, language='english',
                     extra_words=None, keep_words=None):
    """Remove stopwords from a list of tokens."""
    if not isinstance(tokens, list):
        return []
    if not tokens:
        return []

    stop_words = {w.lower() for w in nltk.corpus.stopwords.words(language)}
    if extra_words:
        stop_words.update(w.lower() for w in extra_words)
    if keep_words:
        stop_words.difference_update(w.lower() for w in keep_words)

    return [t for t in tokens
            if not isinstance(t, str) or t.lower() not in stop_words]
