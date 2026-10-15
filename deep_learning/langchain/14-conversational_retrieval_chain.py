#!/usr/bin/env python3
"""builds a retrieval chain capable of answering user queries
by fetching relevant docuemtns while incorporating prior
conversation history"""
from langchain import chains


def create_conversational_retrieval_chain(
        history_aware_retriever, document_chain):
    """builds a retrieval chain capable of answering user queries
    by fetching relevant docuemtns while incorporating prior
    conversation history

    Args:
        history_aware_retriever: The retriever object that takes into account
            the conversation history when fetching relevant documents.
        document_chain: The chain responsible for processing the retrieved
            documents and generating a response.

    Returns:
        A conversational retrieval chain that can answer user queries
        based on relevant documents and prior conversation history.
    """
    conversational_retrieval_chain = chains.ConversationalRetrievalChain(
        retriever=history_aware_retriever,
        combine_docs_chain=document_chain,
        return_source_documents=True
    ).with_config({"run_name": "retrieval_chain"})

    return conversational_retrieval_chain
