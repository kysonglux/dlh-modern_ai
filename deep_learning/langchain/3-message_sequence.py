#!/usr/bin/env python3
"""prepares a sequence of messages combining system message
with one or more human messages
"""
from langchain_core import messages


def setup_message_sequence(system_msg, human_msgs):
    """prepares a sequence of messages combining system message
    with one or more human messages

    Args:
        system_msg (str): The system message to be included in the sequence.
        human_msgs (list):

    Returns:
        A list of messages combining the system message and human messages.
    """
    message_sequence = [messages.SystemMessage(content=system_msg)]
    message_sequence += [(messages.HumanMessage(content=msg)
                          for msg in human_msgs)]
    return message_sequence
