from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from pydantic import BaseModel
from typing import Literal


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="/data/ranjith/lang_graph/week2/chroma_papers_db", embedding_function=embeddings)

@tool
def search_papers(query: str) -> str:
    """Search Ranjith's reference papers about KV-cache divergence and decoding reliability."""
    results = vectorstore.similarity_search(query, k=2)
    if not results:
        return "NO RESULTS FOUND."
    return "\n\n---\n\n".join(f"[page {d.metadata.get('page')}]\n{d.page_content}" for d in results)

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

llm = ChatGroq(model="openai/gpt-oss-120b")

# --- custom state: MessagesState plus a routing field ---
class AgentState(MessagesState):
    next: str

# --- supervisor: structured output forces a valid routing decision ---
class Route(BaseModel):
    next: Literal["research_agent", "math_agent"]

router_llm = llm.with_structured_output(Route)

def supervisor(state: AgentState):
    decision = router_llm.invoke([
        {"role": "system", "content": "Classify the user's question: 'research_agent' for anything about KV-cache, decoding, or research papers; 'math_agent' for arithmetic."},
        state["messages"][-1],
    ])
    return {"next": decision.next}

def research_agent(state: AgentState):
    llm_with_tools = llm.bind_tools([search_papers])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def math_agent(state: AgentState):
    llm_with_tools = llm.bind_tools([add])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def route_decision(state: AgentState) -> Literal["research_agent", "math_agent"]:
    return state["next"]

builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor)
builder.add_node("research_agent", research_agent)
builder.add_node("math_agent", math_agent)
builder.add_node("research_tools", ToolNode([search_papers]))
builder.add_node("math_tools", ToolNode([add]))

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_decision, {"research_agent": "research_agent", "math_agent": "math_agent"})
builder.add_conditional_edges("research_agent", tools_condition, {"tools": "research_tools", END: END})
builder.add_conditional_edges("math_agent", tools_condition, {"tools": "math_tools", END: END})
builder.add_edge("research_tools", "research_agent")
builder.add_edge("math_tools", "math_agent")

graph = builder.compile()

# test both routes
r1 = graph.invoke({"messages": [{"role": "user", "content": "What causes KV-cache divergence?"}]})
print("Research:", r1["messages"][-1].content[:150])

r2 = graph.invoke({"messages": [{"role": "user", "content": "What is 47 + 89?"}]})
print("Math:", r2["messages"][-1].content[:150])

for msg in r1["messages"]:
    print(f"{msg.type}: {getattr(msg, 'tool_calls', None)}")

for i in range(4):
    try:
        r = graph.invoke({"messages": [{"role": "user", "content": "How many pages does my KV-cache paper have?"}]})
        print(f"Trial {i}: {r['messages'][-1].content[:100]}")
    except Exception as e:
        print(f"Trial {i}: ERROR — {str(e)[:150]}")
