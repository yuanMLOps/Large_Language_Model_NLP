from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

# Connect LangGraph to MCP
mcp_client = MultiServerMCPClient(["http://localhost:8000"])  # MCP server URL

tools = await mcp_client.get_tools()
# Define LangGraph agent
agent = create_react_agent(
    tools= tools,  # Fetch tools dynamically
    model="gpt-4",  # LLM model
)

# Example usage
response = agent.run("What is 5 + 3?")
print(response)
