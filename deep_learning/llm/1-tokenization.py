#!/usr/bin/env python3
"""loads a pre-trained RoBERTa tokenizer and tokenizes a given input"""

import transformers


def tokenize_text(model_name, sentence, padding=True):
    """Load a pre-trained RoBERTa tokenizer and tokenize input"""

    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty string.")
    if isinstance(sentence, str):
        sentence = [sentence]
    elif not isinstance(sentence, list) or not all(isinstance(s, str)
                                                   and s for s in sentence):
        raise ValueError("sentence must be a non-empty string.")

    tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)
    inputs = tokenizer(
        sentence,
        padding=padding,
        truncation=True,
        return_tensors="pt"
    )

    return tokenizer, inputs
