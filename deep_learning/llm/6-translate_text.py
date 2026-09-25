#!/usr/bin/env python3
"""creates a high-levl interface for performing language translation"""
import transformers


def translate_text(model_name, src_lang, tgt_lang):
    """ creates a high-level interface for performing language
    translation using a pre-trained large language model."""
    
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty string.")
    if not isinstance(src_lang, str) or not src_lang:
        raise ValueError("src_lang must be a non-empty string.")
    if not isinstance(tgt_lang, str) or not tgt_lang:
        raise ValueError("tgt_lang must be a non-empty string.")

    translator = transformers.pipeline(
        "translation",
        model=model_name,
        tokenizer=model_name,
        src_lang=src_lang,
        tgt_lang=tgt_lang
    )

    return translator