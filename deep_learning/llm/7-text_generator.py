#!/usr/bin/env python3
"""creates a high-level interface for text generation,
allowing control over the genration behavior"""
import transformers


def create_text_generator(model_name, prompt, max_new_tokens,
                          temperature, repetition_penalty,
                          no_repeat_ngram_size):
    """Create a high-level interface for performing text generation"""

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    generator = transformers.pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        pad_token_id=tokenizer.eos_token_id,
    )
    output = generator(prompt, return_full_text=True)
    return generator, output
