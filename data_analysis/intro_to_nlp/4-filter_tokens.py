#!/usr/bin/env python3
"""remove low-information tokens from a list of tokens"""
import re


def filter_tokens(tokens, min_len=2, strip_hashtag=False):
    """Filter tokens based on certain criteria."""
    _PLACEHOLDER_RE = re.compile(r'^<[A-Za-z]+>$')
    if not isinstance(tokens, list) or not tokens:
        return []
    filtered = []
    for t in tokens:
        if not isinstance(t, str):
            raise ValueError("Token must be a string.")
        if _PLACEHOLDER_RE.match(t):
            filtered.append(t)
            continue
        if t.isdigit():
            continue
        if t.isascii() and all(ch in "!?." for ch in t):
            continue
        if min_len is not None and len(t) < min_len:
            continue
        if strip_hashtag:
            t = t.lstrip('#')
        filtered.append(t)
    return filtered
