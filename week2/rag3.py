from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.tools import tool
import glob
import shutil, os


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="/data/ranjith/lang_graph/week2/chroma_papers_db",
    embedding_function=embeddings,
)

# --- TEMPORARY DIAGNOSTIC — remove after checking ---
print("=== Widened search (k=6) ===")
diagnostic_results = vectorstore.similarity_search("KV-cache divergence", k=6)
for r in diagnostic_results:
    print(r.metadata.get("page"), "-", r.page_content[:150])
print("=== end diagnostic ===\n")
# --- end temporary block ---

@tool
def search_papers(query: str) -> str:
    """Search Ranjith's reference papers for passages about KV-cache divergence,
    decoding reliability, or FP16 non-associativity research. Only cite the exact
    source and page numbers shown below — never cite a page that does not appear here."""
    results = vectorstore.similarity_search(query, k=3)
    if not results:
        return "NO RESULTS FOUND. Do not answer this question — tell the user the search returned nothing."
    retrieved_pages = [d.metadata.get("page") for d in results]
    formatted = "\n\n---\n\n".join(
        f"[{d.metadata.get('source')}, page {d.metadata.get('page')}]\n{d.page_content}"
        for d in results
    )
    return f"RETRIEVED PAGES: {retrieved_pages}\n\n{formatted}"

tools = [search_papers]
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

result = graph.invoke({"messages": [{"role": "user", "content": "According to my papers, what causes KV-cache divergence?"}]})
print(result["messages"][-1].content)
print(vectorstore._collection.count())
for msg in result["messages"]:
    print(f"{msg.type}: {getattr(msg, 'tool_calls', None)}")
    print(msg.content[:200])
    print("---")

