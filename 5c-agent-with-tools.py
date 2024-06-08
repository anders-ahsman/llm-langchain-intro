from operator import itemgetter

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

MODEL_NAME = "gpt-4-0613"


@tool
def get_secret(name: str) -> str:
    """Get secret about person."""

    names_to_secrets = {
        "alice": "Alice is a secret agent.",
        "bob": "Bob is a wizard.",
        "charlie": "Charlie is a superhero.",
        "dave": "Dave is a time traveler.",
    }

    return names_to_secrets.get(name.lower(), "I don't know any secrets about that person.")


@tool
def get_employees_for_company(company: str) -> list[str]:
    """Get employees working at company."""

    companies_to_employees = {
        "acme": ["alice", "bob"],
        "ikea": ["dave"],
        "volvo": ["charlie"],
    }

    return companies_to_employees.get(company.lower(), "I don't know about that company.")


def main():
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are very powerful assistant that can help with secrets about people."),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    tools = [get_secret, get_employees_for_company]

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
    )
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
