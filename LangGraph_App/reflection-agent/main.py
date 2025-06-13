from typing import List, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import MessageGraph, END
from chains import generate_chain, reflect_chain


REFLECT = "reflect"
GENERATE = "generate"


def generate_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": state})


def reflect_node(state: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages": state})
    return [HumanMessage(content=res.content)]


# def should_continue(state: List[BaseMessage]):
#     if len(state) > 6:
#         return END
#     return REFLECT


builder = MessageGraph()
builder.add_node(REFLECT, reflect_node)
builder.add_node(GENERATE, generate_node)
builder.set_entry_point(GENERATE)

builder.add_edge(REFLECT, GENERATE)

# you can define a lambda function as condition. The implementation is more concise
# but you will not be able to trace this condition from LangSmith
builder.add_conditional_edges(GENERATE, lambda state: END if len(state)>6 else REFLECT)

# you can use an explicit function to define the condition, which acts as another chain
# this makes the tracing of execution easier to track
# builder.add_conditional_edges(GENERATE, lambda state: END if len(state)>6 else REFLECT)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()

if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = HumanMessage(content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """)
    response = graph.invoke(inputs)