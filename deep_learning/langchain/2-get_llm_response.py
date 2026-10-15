#!/usr/bin/env python3
"""loaded language model and prompt to generate a response"""


def get_response(llm, prompt):
    """generates a response from the language model given a prompt"""

    AIMessage = llm.invoke(prompt)

    return AIMessage
