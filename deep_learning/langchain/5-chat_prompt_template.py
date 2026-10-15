#!/usr/bin/env python3
"""Sets up a reusable chat prompt with placeholders for dynamic content."""
from langchain import prompts


def create_chat_prompt_template(system_message, human_message):
    """Sets up a reusable chat prompt with placeholders for dynamic content.

    Args:
        system_message (str): The system message to be included in the prompt.
        human_message (str): The human message to be included in the prompt.

    Returns:
        A dictionary containing the system and human messages.
    """
    return prompts.ChatPromptTemplate.from_messages(
        [("system", system_message), ("user", human_message)]
    )
