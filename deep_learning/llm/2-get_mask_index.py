#!/usr/bin/env python3
"""returns the positions of all <mask> tokens"""
import transformers


def get_mask_index(inputs, tokenizer):
    """Get the positions of all <mask> tokens in the input tensor.
    """
    if not hasattr(inputs, '__contains__') or 'input_ids' not in inputs:
        raise ValueError("inputs must be a dictionary containing 'input_ids'.")
    if (not hasattr(tokenizer, 'mask_token_id')
            or tokenizer.mask_token_id is None):
        raise ValueError("No <mask> token found in the input!")

    mask_token_id = tokenizer.mask_token_id
    mask_indices = []

    for input_ids in inputs['input_ids']:
        indices = (
                (input_ids == mask_token_id)
                .nonzero(as_tuple=True)[0]
                .tolist()
                )
        mask_indices.append(indices)

    return mask_indices
