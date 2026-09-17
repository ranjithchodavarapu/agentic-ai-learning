from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from pydantic import BaseModel
from typing import Literal
from langchain_ollama import ChatOllama
import re  

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

llm = ChatOllama(model="llama3.1:8b")

class AgentState(MessagesState):
    next: str

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

def extract_text(content):
    """Handles providers that return content as a list of blocks instead of a plain string."""
    if isinstance(content, list):
        return content[0].get("text", "") if content and isinstance(content[0], dict) else str(content)
    return content

print("=== Single test call ===", flush=True)
try:
    r = graph.invoke({"messages": [{"role": "user", "content": "What is 47 + 89?"}]})
    print("SUCCESS:", extract_text(r["messages"][-1].content), flush=True)
except Exception as e:
    print(f"FULL ERROR:\n{e}\n", flush=True)
print("=== end diagnostic ===\n", flush=True)

N_TRIALS = 10

capability_questions = [
    {"question": "What causes KV-cache divergence?", "expected_tool": "research", "answerable_by_tool": True},
    {"question": "How does FP32 affect divergence?", "expected_tool": "research", "answerable_by_tool": True},
    {"question": "How many pages is my KV-cache paper?", "expected_tool": "research", "answerable_by_tool": False},
    {"question": "What year was my KV-cache paper published?", "expected_tool": "research", "answerable_by_tool": False},
    {"question": "Who are the co-authors on my KV-cache paper?", "expected_tool": "research", "answerable_by_tool": False},
    {"question": "What is 47 + 89?", "expected_tool": "math", "answerable_by_tool": True},
]

def run_capability_eval(questions, n_trials=N_TRIALS):
    results = []
    for q in questions:
        for trial in range(n_trials):
            print(f"Running: {q['question'][:40]} (trial {trial})...", flush=True)
            try:
                r = graph.invoke({"messages": [{"role": "user", "content": q["question"]}]})
                answer = extract_text(r["messages"][-1].content)
                declined = any(phrase in answer.lower() for phrase in [
                    "don't have", "cannot determine", "not available", "no information",
                    "unable to", "doesn't specify", "not specified", "don't know",
                ])
                results.append({
                    "question": q["question"][:40],
                    "trial": trial,
                    "answerable_by_tool": q["answerable_by_tool"],
                    "declined": declined,
                    "answer_preview": answer[:120],
                })
            except Exception as e:
                results.append({
                    "question": q["question"][:40],
                    "trial": trial,
                    "error": str(e)[:200],
                    "answerable_by_tool": q["answerable_by_tool"],
                })
    return results

results = run_capability_eval(capability_questions, n_trials=N_TRIALS)
for r in results:
    print(r, flush=True)

incapable_questions = [q for q in capability_questions if not q["answerable_by_tool"]]
incapable_attempted = len(incapable_questions) * N_TRIALS
incapable_succeeded = [r for r in results if "error" not in r and not r.get("answerable_by_tool")]
if not incapable_succeeded:
    print(f"\nCorrect-decline rate: N/A — 0/{incapable_attempted} succeeded")
else:
    rate = sum(r["declined"] for r in incapable_succeeded) / len(incapable_succeeded)
    print(f"\nCorrect-decline rate: {rate:.0%} (n={len(incapable_succeeded)}/{incapable_attempted})")