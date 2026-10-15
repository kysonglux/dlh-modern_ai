#!/usr/bin/env python3
"""combines multiple documents into coherent context and generates answers"""
from langchain_core import prompts
from langchain.chains import combine_documents


def document_answer_chain(llm, system_message, human_message):
    """combines multiple documents into coherent context and generates answers
    """
    prompt = prompts.ChatPromptTemplate.from_messages([
        ("system", system_message),
        prompts.MessagesPlaceholder(variable_name="chat_history"),
        ("human", human_message)
    ])

    document_chain = (combine_documents.
                      create_stuff_documents_chain(llm, prompt))

    return prompt, document_chain
