#!/usr/bin/env python3
"""compute the raw logits for all <mask> tokens
in a given input using a pre-trained RoBERTa model"""
import torch


def compute_mask_logits(model, inputs, mask_indices):
    """Compute <mask> tokens in a given input
    using a pre-trained RoBERTa model.
    """
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    mask_logits_list = [logits[0, idx] for idx in mask_indices]

    return mask_logits_list
