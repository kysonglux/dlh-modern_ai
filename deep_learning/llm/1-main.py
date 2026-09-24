#!/usr/bin/env python3

tokenize_text = __import__('1-tokenization').tokenize_text

sentences = [
    "AI is dominating the world.",
    "Transformers are the backbone of modern LLMs.",
    "Hello!"
]

tokenizer, inputs = tokenize_text("roberta-base", sentences)

print("Tokenizer type:", type(tokenizer))
print("Input IDs:", inputs['input_ids'])
print("Tokens:")
for i, ids in enumerate(inputs['input_ids']):
    print(f"Sentence {i+1}:", tokenizer.convert_ids_to_tokens(ids))
print("Attention mask:", inputs['attention_mask'])