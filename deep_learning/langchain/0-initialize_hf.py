#!/usr/bin/env python3
"""sets up a HF text to text generation model using langchain"""
import transformers
from langchain_community.llms import HuggingFacePipeline


def initialize_hf_llm(model_name, max_tokens):
    """initializes a HF text to text generation model using langchain"""

    # Create a text generation pipeline using the specified model
    text_gen_pipeline = transformers.pipeline(
            "text-generation",
            model=model_name)

    # Wrap the pipeline in a HuggingFacePipeline object for use with LangChain
    llm = HuggingFacePipeline(pipeline=text_gen_pipeline,
                              model_kwargs={"max_length": max_tokens})

    return llm
