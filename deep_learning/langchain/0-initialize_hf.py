#!/usr/bin/env python3
"""sets up a HF text to text generation model using langchain"""
from langchain_community import llms


def initialize_hf_llm(model_name, max_tokens):
    """initializes a HF text to text generation model using langchain"""

    llm = llms.HuggingFacePipeline.from_model_id(
        model_id=model_name,
        task="text2text-generation",
        pipeline_kwargs={"max_new_tokens": max_tokens})

    return llm
