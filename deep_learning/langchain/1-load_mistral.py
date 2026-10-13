#!/usr/bin/env python3
"""initialize a mistral language model using langchain"""
from langchain_mistralai import ChatMistralAI


def load_mistral(model_name, temperature):
    """initializes a mistral language model using langchain"""

    llm = ChatMistralAI(
        model=model_name,
        temperature=temperature)

    return llm
