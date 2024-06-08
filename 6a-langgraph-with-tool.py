from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessageGraph
from langgraph.prebuilt import ToolNode


@tool
def multiply(first_number: int, second_number: int):
    """Multiplies two numbers together."""
    return first_number * second_number


model = ChatOpenAI(temperature=0)
model_with_tools = model.bind_tools(tools=[multiply])

graph = MessageGraph()

graph.add_node("oracle", model_with_tools)

tool_node = ToolNode([multiply])
graph.add_node("multiply", tool_node)
graph.add_edge(START, "oracle")
graph.add_edge("multiply", END)


def router(state: list[BaseMessage]) -> Literal["multiply", "__end__"]:
    tool_calls = state[-1].additional_kwargs.get("tool_calls", [])
    return "multiply" if len(tool_calls) else END


graph.add_conditional_edges("oracle", router)
runnable = graph.compile()

while True:
    user_input = input("Ask a question (or type 'exit'): ")
    if user_input == "exit":
        break

    response = runnable.invoke(HumanMessage(user_input))
    print(response)
