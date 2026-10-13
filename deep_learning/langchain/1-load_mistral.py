#!/usr/bin/env python3
"""initialize a mistral language model using langchain"""
import langchain_mistralai


def load_mistral(model_name, temperature):
    """initializes a mistral language model using langchain"""

    llm = langchain_mistralai.ChatMistralAI(
        model=model_name,
        temperature=temperature)

    return llm
