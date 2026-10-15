#!/usr/bin/env python3
"""sets up a Pandas DataFrame Agent using LangChain LLMs"""
from langchain_experimental.agents.agent_toolkits import (
    create_pandas_dataframe_agent)


def create_dataframe_agent(llm, df):
    """sets up a Pandas DataFrame Agent using LangChain LLMs

    Args:
        llm: The language model to be used for the agent.
        df: The Pandas DataFrame to be used for the agent.

    Returns:
        A Pandas DataFrame Agent.
    """

    return create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        agent_executor_kwargs={"return_intermediate_steps": True})
