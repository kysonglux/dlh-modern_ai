#!/usr/bin/env python3
"""builds a FAISS vector store from a list of documents using LangChain"""
from langchain.vectorstores import FAISS


def create_faiss_vectorstore(docs, embedding_model,
                             k, search_type="similarity"):
    """builds a FAISS vector store from a list of documents using LangChain
    """

    vectorstore = FAISS.from_documents(
        docs,
        embedding_model,
        k=k,
        search_type=search_type
    )
    return vectorstore
