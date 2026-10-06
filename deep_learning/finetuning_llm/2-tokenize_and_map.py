#!/usr/bin/env python3
"""tokenize the text inputs of the entire Emotion dataset"""


def tokenize_and_map(dataset, tokenizer, max_length, truncation, batched):
    """
    Tokenize the text inputs of the entire Emotion dataset.
    """
    def tokenize_function(examples):
        """tokenized function"""
        return tokenizer(examples["text"],
                         truncation=truncation,
                         max_length=max_length)

    tokenized_datasets = dataset.map(tokenize_function, batched=batched)

    return (tokenized_datasets["train"], tokenized_datasets["validation"],
            tokenized_datasets["test"])
