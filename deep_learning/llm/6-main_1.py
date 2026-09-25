#!/usr/bin/env python3

translate_text = __import__('6-translate_text').translate_text

translator_en_fr = translate_text("Helsinki-NLP/opus-mt-en-fr")
sentence1 = "Artificial intelligence is transforming the world."
sentence2 = "Machine learning models improve over time."

translations = translator_en_fr([sentence1, sentence2])
for i, t in enumerate(translations, 1):
    print(f"Sentence {i} translation:", t['translation_text'])