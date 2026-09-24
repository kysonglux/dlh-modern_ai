#!/usr/bin/env python3
"""builds a TF-IDF feature matrix from a list of tokens"""
import sklearn


def tf_idf(corpus_tokens, max_features=5000,
           ngram_range=(1, 2), min_df=2, max_df=0.95, norm='l2'):
    """Builds a TF-IDF feature matrix from a list of tokens."""
    if not isinstance(corpus_tokens, list) or not corpus_tokens:
        return
    if not isinstance(max_features, int) or max_features < 1:
        raise ValueError("max_features must be a positive integer.")
    if not isinstance(ngram_range, tuple) or len(ngram_range) != 2:
        raise ValueError("ngram_range must be a tuple of length 2.")
    if not isinstance(min_df, (int, float)) or min_df < 0:
        raise ValueError("min_df must be a non-negative integer or float.")
    if not isinstance(max_df, (int, float)) or max_df <= 0:
        raise ValueError("max_df must be a positive integer or float.")
    if not isinstance(norm, str) or norm not in ['l1', 'l2', 'max']:
        raise ValueError("norm must be one of 'l1', 'l2', or 'max'.")
    documents = [" ".join(str(t) for t in tokens) for tokens in corpus_tokens]
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        norm=norm,
        tokenizer=str.split,
        lowercase=False,
        token_pattern=None,
    )
    x = vectorizer.fit_transform(documents)
    return x, vectorizer
