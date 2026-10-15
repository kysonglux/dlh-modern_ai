#!/usr/bin/env python3
"""generates a serach query based on the conversation history
and the user's current input to retrieve the most relevant documents"""
from lanchiain_core import prompts
from langchain.chains import chains


def build_history_aware_retriever(llm, retriever, human_message):
    """generates a serach query based on the conversation history
    and the user's current input to retrieve the most relevant documents

    Args:
        llm: The language model to be used for generating the search query.
        retriever: The retriever object used to fetch relevant documents.
        human_message: The current input from the user.

    Returns:
        A list of retrieved documents based on the generated search query.
    """
    prompt = prompts.ChatPromptTemplate.from_messages([
        prompts.MessagesPlaceholder(variable_name="chat_history"),
        ("human", human_message)
    ])

    history_aware_retriever = chains.create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=prompt
    )

    return prompt, history_aware_retriever
