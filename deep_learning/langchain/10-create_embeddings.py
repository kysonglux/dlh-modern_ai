#!/usr/bin/env python3
"""loads a HF embedding model using LangChain"""
from langchain.embeddings import HuggingFaceEmbeddings


def create_embeddings(model_name):
    """loads a HF embedding model using LangChain

    Args:
        model_name (str): The name of the HF embedding model to be loaded.

    Returns:
        An instance of the HF embedding model.
    """

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
