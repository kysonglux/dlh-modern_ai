#!/usr/bin/env python3

tokenize_text = __import__('1-tokenization').tokenize_text
get_mask_index = __import__('2-get_mask_index').get_mask_index

sentences = [
    "AI will <mask> the future and <mask> the world.",
    "Transformers are <mask> modern LLMs."
]

for i, sentence in enumerate(sentences, 1):
    print(f"\nSentence {i}: {sentence}")

    tokenizer, inputs = tokenize_text("roberta-base", sentence, padding=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    token_ids = inputs["input_ids"][0]

    print("Tokens:", tokens)
    print("Token IDs:", token_ids)

    mask_indices = get_mask_index(inputs, tokenizer)
    print("Mask indices:", mask_indices)