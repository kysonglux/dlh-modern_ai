#!/usr/bin/env python3
"""sets up a resuable prompt with placeholders for dynamic content"""
from langchain import prompts


def create_prompt_template(template_str, input_variables):
    """sets up a resuable prompt with placeholders for dynamic content"""

    return prompts.PromptTemplate(
        template=template_str,
        input_variables=input_variables
    )
