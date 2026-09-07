from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

@tool
def search_stub(query: str) -> str:
    """Pretend to search the web for a query"""
    return f"Fake search results for: {query}"

tools = [add, search_stub]

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

graph = builder.compile()

#result = graph.invoke({"messages": [{"role": "user", "content": "What is 5 + 7?"}]})
##print(result["messages"][-1].content)

result = graph.invoke({"messages": [{"role": "user", "content": "search for langgraph tutorials"}]})
for msg in result["messages"]:
    print(f"{msg.type}: {msg.content}")