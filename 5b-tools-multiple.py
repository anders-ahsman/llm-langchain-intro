from operator import itemgetter
from typing import Any

from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
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
    tools = [add, multiply]
    tool_map = {tool.name: tool for tool in tools}

    llm = ChatOpenAI(
        model="gpt-3.5-turbo-1106",
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(tools)

    def call_tool(tool_invocation: dict) -> Runnable:
        """Function for dynamically constructing the end of the chain based on the model-selected tool."""
        tool = tool_map[tool_invocation["type"]]
        return RunnablePassthrough.assign(output=itemgetter("args") | tool)

    def get_first_output(responses: list[dict]) -> Any:
        return responses[0]["output"]

    call_tool_list = RunnableLambda(call_tool).map()
    tool_chain = JsonOutputToolsParser() | call_tool_list | RunnableLambda(get_first_output)

    # note: this handles multiple tools, but only one tool will be called (or none)
    # e.g. "what is one plus two times three?" does not work

    while True:
        user_input = input("Ask a question (or type 'exit'): ")
        if user_input == "exit":
            return

        response = llm_with_tools.invoke(user_input)
        if 'tool_calls' in response.additional_kwargs:
            # LLM generated arguments for function call, use that to call the tool
            tool_response = tool_chain.invoke(response)
            print(tool_response)
        else:
            # LLM decided not to call any tool, print normal response
            print(response.content)


if __name__ == "__main__":
    main()
