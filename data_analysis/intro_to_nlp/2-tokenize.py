#!/usr/bin/env python3
"""tokenize.py — Tokenize text for NLP tasks."""
import nltk


EMOTICON_MAP = {
    "<3":   "<EMO>", "</3": "<EMO>",
    ":)":   "<EMO>", ":-)": "<EMO>",
    ":(":   "<EMO>", ":-(": "<EMO>",
    ":d":   "<EMO>", ";)":  "<EMO>",
    ":|":   "<EMO>", ">:(": "<EMO>",
    ":p":   "<EMO>", "b)":  "<EMO>",
    "o:)":  "<EMO>",
}


def normalize_emoticons(tokens, emoticon_action="replace"):
    """Normalize emoticons in a list of tokens."""
    if not isinstance(tokens, list):
        return []

    result = []

    for token in tokens:
        mapped = EMOTICON_MAP.get(token.lower())

        if mapped:
            if emoticon_action == "replace":
                result.append(mapped)
        else:
            result.append(token)

    return result


def tokenize_text(text, method="tweet"):
    """Tokenize text using the specified method."""
    if not isinstance(text, str):
        return []

    if method == "tweet":
        return nltk.word_tokenize(text)
    elif method == "word":
        return text.split()
    elif method == "split":
        return text.split()
    else:
        return []
