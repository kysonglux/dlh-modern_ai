#!/usr/bin/env python3
"""sets up a langchain agent executor capable of
using multiple tools to answer user queries"""
from langchain import agents
from langchain_community.agent_toolkits import load_tools


def load_agent_with_toolkit(llm, toolkit_names, prompt):
    """sets up a langchain agent executor capable of
    using multiple tools to answer user queries
    """

    tools = load_tools(toolkit_names, llm=llm)
    agent = agents.create_tool_calling_agent(llm, tools, prompt)
    agent_executor = agents.AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )
    return agent_executor
