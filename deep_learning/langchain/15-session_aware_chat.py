#!/usr/bin/env python3
"""wraps a conversational retrieval chain to support
session-aware chat histories. """
from langchain_core import runnables


def create_session_aware_chat(retrieval_chain, get_session_history,
                              input_key, history_key, output_key):
    """wraps a conversational retrieval chain to support
    session-aware chat histories.
    """
    return (runnables.history.RunnableWithMessageHistory(
            retrieval_chain,
            get_session_history,
            input_message_key=input_key,
            history_messages_key=history_key,
            output_message_key=output_key))
