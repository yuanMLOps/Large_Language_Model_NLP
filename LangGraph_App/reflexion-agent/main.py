from typing import List

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph

from chains import revisor, first_responder
from tool_executor import execute_tools

MAX_ITERATIONS = 2
builder = MessageGraph()

# in this project, we directly used chain/Runnable objects
# to build graph nodes. Compared to reflection-agent, where
# we use the functions that invokes chain objects. Both will work
# using functions provides more flexibility, but using chains
# are more concise and compact.

builder.add_node("draft", first_responder)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revisor)

builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")


def should_continue(state: List[BaseMessage]) -> str:
    count_tool_use = sum(isinstance(item, ToolMessage) for item in state)

    if count_tool_use > MAX_ITERATIONS:
        return END
    return "execute_tools"


builder.add_conditional_edges("revise", should_continue)
builder.set_entry_point("draft")
graph = builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path="graph.png")


if __name__ == "__main__":
    print("hello world")

    res = graph.invoke(
        "Write about AI-Powered SOC / autonomous soc  problem domain, list startups that do that and raised capital."
    )
    print(res[-1].tool_calls[0]["args"]["answer"])
    print(res)
