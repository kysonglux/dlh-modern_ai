#!/usr/bin/env python3
"""converts the logits for all <mask> tokens
into their corresponding vocabulary tokens."""


def decode_mask_predictions(mask_logits_list, tokenizer):
    """Convert the logits for all <mask> tokens"""

    decoded_tokens = []
    for mask_logits in mask_logits_list:
        vocab_size = mask_logits.shape[-1]
        token_ids = list(range(vocab_size))
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        decoded_tokens.append(tokens)

    return decoded_tokens
