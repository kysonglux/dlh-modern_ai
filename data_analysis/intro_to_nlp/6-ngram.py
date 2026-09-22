#!/usr/bin/env python3
"""generates n-grams from a list of tokens"""
import nltk


def generate_ngrams(tokens, n=2):
    """Generates n-grams from a list of tokens."""
    if not isinstance(tokens, list) or not tokens:
        return []
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer.")
    if n == 1:
        return tokens
    return list(map("_".join, nltk.ngrams(tokens, n)))
