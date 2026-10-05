#!/usr/bin/env python3
"""dynamic padding for the Emotion dataset"""
import transformers


def create_data_collator(tokenizer):
    """
    Create a data collator for dynamic padding of the Emotion dataset.
    """

    return transformers.DataCollatorWithPadding(tokenizer=tokenizer)
