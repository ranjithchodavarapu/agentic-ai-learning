from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
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

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "test-1"}}

print("=== First question (streamed) ===")
for event in graph.stream({"messages": [{"role": "user", "content": "What is 5 + 7?"}]}, config): # get step-by step events
    for node_name, node_output in event.items():
        print(f"--- {node_name} ---")
        print(node_output)

config3 = {"configurable": {"thread_id": "test-2"}}
for event in graph.stream({"messages": [{"role": "user", "content": "what is 84729 times 5013?"}]}, config3):
    for node_name, node_output in event.items():
        print(f"--- {node_name} ---")
        print(node_output)

print("\n=== Inspecting saved state ===")
state = graph.get_state(config)
print("Message count:", len(state.values["messages"]))
print("Next node to run:", state.next)

print("\n=== Second question, same thread (tests memory) ===")
result2 = graph.invoke({"messages": [{"role": "user", "content": "what was my previous question?"}]}, config)
print(result2["messages"][-1].content)