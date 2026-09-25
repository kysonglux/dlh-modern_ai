#!/usr/bin/env python3
"""creates a high-level interface for text generation,
allowing control over the genration behavior"""
import transformers


def create_text_generator(model_name, prompt, max_new_tokens,
                          temperature, repetition_penalty,
                          no_repeat_ngram_size):
    """Create a high-level interface for performing text generation"""
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty string.")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string.")
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer.")
    if not isinstance(temperature, (int, float)) or temperature <= 0:
        raise ValueError("temperature must be a positive number.")
    if (not isinstance(repetition_penalty, (int, float))
       or repetition_penalty <= 0):
        raise ValueError("repetition_penalty must be a positive number.")
    if not isinstance(no_repeat_ngram_size, int) or no_repeat_ngram_size < 0:
        raise ValueError(" must be a non-negative integer.")

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
