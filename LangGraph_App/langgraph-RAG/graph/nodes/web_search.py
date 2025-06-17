from typing import Any, Dict
from typing_extensions import Annotated

from langchain.schema import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path("../../../../.env"))
web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]

    tavily_results = web_search_tool.invoke({"query": question})
    joined_tavily_result ="\n".join([tavily_result["content"] for tavily_result in tavily_results['results']])

    web_results = Document(page_content=joined_tavily_result)
    if "documents" in state:
        documents = state["documents"].append(web_results)
    else:
        documents = [web_results]

    return {"documents": documents, "question": question}

if __name__ == "__main__":
    web_search(state={"question": "how to make pizza", "documents": None})
