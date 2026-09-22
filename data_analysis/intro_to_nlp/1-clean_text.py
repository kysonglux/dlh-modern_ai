#!/usr/bin/env python3
"""Clean and normalises SMS messages"""
import re
import emoji


_DATASET_PLACEHOLDER_MAP = {
    '<#>':       '<NUM>',
    '<decimal>': '<NUM>',
    '<time>':    '<TIME>',
    '<url>':     '<URL>',
    '<email>':   '<EMAIL>',
}


def normalize_unicode_punct(text):
    """Replace curly quotes, dashes, ellipses, etc. with ASCII equivalents."""
    replacements = {
        '[\u2018\u2019\u201a\u201b]': "'",
        '[\u201c\u201d\u201e\u201f]': '"',
        '[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]': '-',
        '\u2026': '...'
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    return text


def clean_text(text, replace_num=True,
               replace_url=True, emoji_action="replace"):
    """Clean text for NLP tasks."""
    # 1. lowercase + strip
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    # 2. dataset placeholders
    for old, new in _DATASET_PLACEHOLDER_MAP.items():
        text = text.replace(old, new)
    # 3. normalize_unicode_punct()
    text = normalize_unicode_punct(text)
    # 4. URL replacement
    if replace_url:
        text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)
    # 5. number replacement (2 passes)
    if replace_num:
        # Phone numbers
        text = re.sub(r'\+?\d[\d\s\-]{6,}\d', "<NUM>", text)
        # integers, decimas, currency
        text = re.sub(r'(?:£|\$|€)\d+(?:[.,]\d+)*|(?<!<)\b\d+(?:[.,]\d+)*\b',
                      "<NUM>", text)
        # emoticons
        text = re.sub(r'(?<![A-Za-z])\d+(?![A-Za-z])', "<NUM>", text)
    # 6. emoji handling
    if emoji_action == "replace":
        text = emoji.replace_emoji(text, replace='<EMO>')
    elif emoji_action == "remove":
        text = emoji.replace_emoji(text, replace='')
    # 7. collapse repeated ! / ?
    text = re.sub(r'([!?])\1+', r'\1', text)
    # 8. collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
