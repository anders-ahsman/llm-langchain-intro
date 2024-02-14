from langchain.output_parsers import JsonOutputKeyToolsParser
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together."""
    return a * b


def main():
    llm = ChatOpenAI()

    # setup LLM with multiply tool (function) available
    llm_with_tool = llm.bind_tools([multiply])

    # setup chain for parsing LLM output and calling multiply tool
    tool_chain = JsonOutputKeyToolsParser(key_name="multiply", return_single=True) | multiply

    while True:
        user_input = input("Ask a question (or type 'exit'): ")
        if user_input == "exit":
            return

        response = llm_with_tool.invoke(user_input)
        if 'tool_calls' in response.additional_kwargs:
            # LLM generated arguments for function call, use that to call the tool
            tool_response = tool_chain.invoke(response)
            print(tool_response)
        else:
            # LLM decided not to call the tool, print normal response
            print(response.content)


if __name__ == "__main__":
    main()
