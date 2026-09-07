import asyncio
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

@tool
def search_stub(query: str) -> str:
    """Pretend to search the web for a query"""
    return f"Fake search results for: {query}"

async def main():
    client = MultiServerMCPClient({
        "mini_project": {
            "command": "python3",
            "args": ["/data/ranjith/lang_graph/mcp_server.py"],  # use the full absolute path
            "transport": "stdio",
        }
    })
    mcp_tools = await client.get_tools()   # loads "add" from the running MCP server

    tools = mcp_tools + [search_stub]      # mix MCP tools and local tools freely

    llm = ChatGroq(model="openai/gpt-oss-120b")
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: MessagesState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "mcp-test"}}
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "what is 47 + 89?"}]}, config
    )
    print(result["messages"][-1].content)

asyncio.run(main())