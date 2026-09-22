#!/usr/bin/env python3
"""remove low-information tokens from a list of tokens"""
import re


def filter_tokens(tokens, min_len=2, strip_hashtags=False):
    """Filter tokens based on certain criteria."""
    _PLACEHOLDER_RE = re.compile(r'^\[A-Za-z]+>$')
    if not isinstance(tokens, list) or not tokens:
        return []
    filtered = [t for t in tokens]
    for token in filtered:
        if not isinstance(token, str):
            raise ValueError("Token must be a string.")
        if _PLACEHOLDER_RE.match(token):
            filtered = [t for t in filtered if not _PLACEHOLDER_RE.match(t)]
        if min_len is not None:
            filtered = [t for t in filtered if len(t) >= min_len]
        if strip_hashtags:
            filtered = [t.lstrip('#') for t in filtered]
    return filtered
