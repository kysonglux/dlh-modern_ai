#!/usr/bin/env python3
"""creates a high-level interface for performing Masked language Modeling
using a pre-trained large language model"""
import transformers


def fill_mask(model_name, top_k):
    """Create a high-level interface"""

    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty string.")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    fill = transformers.pipeline(
        "fill-mask",
        model=model_name,
        tokenizer=model_name,
        top_k=top_k
    )
    return fill
