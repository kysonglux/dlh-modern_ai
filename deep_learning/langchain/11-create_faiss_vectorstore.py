#!/usr/bin/env python3
"""builds a FAISS vector store from a list of documents using LangChain"""
from langchain_community import vectorstores


def create_faiss_vectorstore(docs, embedding_model,
                             k, search_type="similarity"):
    """builds a FAISS vector store from a list of documents using LangChain
    """

    vectorstore = vectorstores.FAISS.from_documents(
        documents=docs,
        embedding=embedding_model
    )
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )
    return vectorstore, retriever
