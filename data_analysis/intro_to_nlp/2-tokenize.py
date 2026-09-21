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
    """Tokenize text without using TweetTokenizer."""
    if not isinstance(text, str):
        return []

    tokens = []
    if method == "tweet":
        tokenizer = nltk.TweetTokenizer(
            preserve_case=True,
            reduce_len=True,
            strip_handles=False, )
        tokens = tokenizer.tokenize(text)
    elif method == "word":
        tokens = nltk.word_tokenize(text)
    elif method == "split":
        tokens = text.split()
    else:
        raise ValueError("Invalid tokenizer method")
    return tokens
