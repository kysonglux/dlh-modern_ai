#!/usr/bin/env python3
"""creates a high-level interface for performing language translation"""
import transformers


def translate_text(model_name, src_lang="en", tgt_lang="fr"):
    """ creates a high-level interface for performing language
    translation using a pre-trained large language model."""

    translator = transformers.pipeline(
        "translation",
        model=model_name,
        tokenizer=model_name,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )

    return translator
