#!/usr/bin/env python3
"""loads documents from a folder and returns them as a list using langchain"""
from langchain_community.download_loader import DirectoryLoader, TextLoader


def load_documents(folder_path, file_pattern):
    """loads documents from a folder and returns them as a list using langchain

    Args:
        folder_path (str): The path to the folder containing the documents.
        file_pattern (str): The pattern to match files in the folder.

    Returns:
        A list of loaded documents.
    """

    loader = DirectoryLoader(folder_path,
                             glob=file_pattern,
                             loader_cls=TextLoader)
    documents = loader.load()
    return documents
