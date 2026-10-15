#!/usr/bin/env python3
"""sets up a langchain agent executor capable of
using multiple tools to answer user queries"""
from langchain import agents


def load_agent_with_toolkit(llm, toolkit_names, prompt):
    """sets up a langchain agent executor capable of
    using multiple tools to answer user queries
    """

    tools = agents.load_tools(toolkit_names, llm=llm)
    agent = agents.creaate_tool_calling_agentt(llm, tools, prompt)
    agent_executor = agents.AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )
    return agent_executor
