#!/usr/bin/env python3
"""combines multiple documents into coherent context and generates answers"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain


def document_answer_chain(llm, system_message, human_message):
    """combines multiple documents into coherent context and generates answers
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", human_message)
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)

    return prompt, document_chain
