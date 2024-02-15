from operator import itemgetter

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def add(a: int, b: int) -> int:
    """Add two integers together."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together."""
    return a * b


def main():
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are very powerful assistant, but don't know current events"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    tools = [add, multiply]

    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    # NOTE: this agent does not have any memory of previous calls, so conversation won't work
    agent = (
        {
            "input": itemgetter("input"),
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools)  # use verbose=True to see intermediate steps

    while True:
        user_input = input("Ask a question (or type 'exit'): ")
        if user_input == "exit":
            return

        response = agent_executor.invoke({"input": user_input})
        print(response["output"])


if __name__ == "__main__":
    main()
