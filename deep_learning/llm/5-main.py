#!/usr/bin/env python3

from pprint import pprint
fill_mask = __import__('5-fill_mask').fill_mask

sentences = [
    "AI can <mask> productivity, <mask> creativity, and even <mask> decision-making.",
    "Machine learning models can <mask> patterns in data and <mask> accurate predictions.",
    "Climate change is <mask> the planet and <mask> future generations."
]

fill = fill_mask(model_name="roberta-base", top_k=1)

for i, sentence in enumerate(sentences, 1):
    print(f"\nSentence {i}: {sentence}")
    predictions = fill(sentence)
    pprint(predictions)