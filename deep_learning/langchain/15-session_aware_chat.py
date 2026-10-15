#!/usr/bin/env python3
"""wraps a conversational retrieval chain to support
session-aware chat histories. """
from langchain-core import RunnablewWithMessageHistory


def create_session_aware_chat(retrieval_chain, get_session_history,
                              input_key, history_key, output_key):
    """wraps a conversational retrieval chain to support
    session-aware chat histories.
    """
    return (RunnablewWithMessageHistory(
            retrieval_chain,
            get_session_history, input_key, history_key, output_key))
