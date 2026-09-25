#!/usr/bin/env python3
"""converts the logits for all <mask> tokens
into their corresponding vocabulary tokens."""


def decode_mask_predictions(mask_logits_list, tokenizer):
    """Convert the logits for all <mask> tokens"""

    decoded_tokens = []
    for mask_logits in mask_logits_list:
        top_indexs = mask_logits.argmax(dim=-1).item()
        tokens = tokenizer.decode([top_indexs]).strip()
        decoded_tokens.append(tokens)

    return decoded_tokens
