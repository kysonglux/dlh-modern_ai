#!/usr/bin/env python3
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
        r"[''‚‛]":    "'",
        r"[""„‟]":    '"',
        r"[‐‑‒–—―−]": "-",
        r"…":          "...",
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    return text


def clean_text(text, replace_num=True,
               replace_url=True, emoji_action="replace"):
    """Clean text for NLP tasks."""
    # 1. lowercase + strip
    if not text:
        return ''
    text = text.lower().strip()
    # 2. dataset placeholders
    for old, new in _DATASET_PLACEHOLDER_MAP.items():
        text = text.replace(old, new)
    # 3. normalize_unicode_punct()
    text = normalize_unicode_punct(text)
    # 4. URL replacement
    if replace_url:
        text = re.sub(r'http\S+|www\S+', '<URL>', text)
    # 5. number replacement (2 passes)
    if replace_num:
        text = re.sub(r'\d+\.\d+', '<NUM>', text)
        text = re.sub(r'\d+', '<NUM>', text)
    # 6. emoji handling
    if emoji_action == "replace":
        text = emoji.emojize(text, language='en')
    elif emoji_action == "remove":
        text = emoji.demojize(text)
    # 7. collapse repeated ! / ?
    text = re.sub(r'([!?])\1+', r'\1', text)
    # 8. collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
