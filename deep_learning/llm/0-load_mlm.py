#!/usr/bin/env python3
"""Load a pre-trained Masked Language Model (MLM)."""
from transformers import RobertaForMaskedLM


def load_mlm(model_name):
    """Load a pre-trained RoBERTa model ready
    for Masked Language Modeling (MLM) using Hugging Face Transformers."""

    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty string.")

    model = RobertaForMaskedLM.from_pretrained(model_name)

    return model
