#!/usr/bin/env python3
"""returns the positions of all <mask> tokens"""


def get_mask_index(inputs, tokenizer):
    """Get the positions of all <mask> tokens in the input tensor.
    """

    mask_token_id = tokenizer.mask_token_id
    input_ids = inputs['input_ids'][0]

    mask_indices = (
                (input_ids == mask_token_id)
                .nonzero(as_tuple=True)[0]
                .tolist()
                )

    if not mask_indices:
        raise ValueError("No <mask> token found in the input!")

    return mask_indices
