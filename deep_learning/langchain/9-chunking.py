#!/usr/bin/env python3
"""splits a list of documents into smaller, managebale chunks"""
from langchain import text_splitter


def split_into_chunks(docs, chunk_size, chunk_overlap):
    """splits a list of documents into smaller, managebale chunks

    Args:
        docs (list): A list of documents to be split into chunks.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of overlapping characters.

    Returns:
        A list of document chunks.
    """

    splitter = text_splitter.RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    list = splitter.split_documents(docs)
    return list
