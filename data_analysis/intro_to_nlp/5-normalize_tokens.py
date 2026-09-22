#!/usr/bin/env python3
"""normalises tokens via lemmatisation or stemming"""
import nltk
import re


PLACEHOLDER_PATTERN = re.compile(r"^(<[A-Z0-9_]+>|__[A-Z0-9_]+__)$")


def is_placeholder(token):
    """Returns True
    if the token is a placeholder that should not be normalized."""
    return bool(PLACEHOLDER_PATTERN.match(token))


def get_pos(tag):
    """Maps a Penn Treebank POS tag to a WordNet POS tag."""
    if tag.startswith("J"):
        return nltk.corpus.wordnet.ADJ
    elif tag.startswith("V"):
        return nltk.corpus.wordnet.VERB
    elif tag.startswith("R"):
        return nltk.corpus.wordnet.ADV
    else:
        # Default to noun (covers NN*, and anything else unmapped)
        return nltk.corpus.wordnet.NOUN


def normalize_tokens(tokens, method="lemmatize"):
    """Normalizes tokens using the specified method."""
    if not isinstance(tokens, list) or not tokens:
        return []
    if not isinstance(method, str) or method not in {"lemmatize", "stem"}:
        raise ValueError("method must be 'lemmatize' or 'stem'")
    normalized = []
    if method == "stem":
        stemmer = nltk.PorterStemmer()
        for t in tokens:
            if is_placeholder(t):
                normalized.append(t)
            else:
                normalized.append(stemmer.stem(t))
    else:  # method == "lemmatize"
        lemmatizer = nltk.WordNetLemmatizer()
        tagged_tokens = nltk.pos_tag(tokens)
        for t, tag in tagged_tokens:
            if is_placeholder(t):
                normalized.append(t)
            else:
                normalized.append(lemmatizer.lemmatize(t, get_pos(tag)))
    return normalized
