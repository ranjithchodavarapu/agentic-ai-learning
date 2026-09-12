from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.tools import tool
import glob
import shutil, os
import re   

def check_citation_grounding(answer: str, retrieved_pages: list[int]) -> dict:
    #cited_pages = [int(p) for p in re.findall(r"\[PAGE:(\d+)\]", answer)]
    #cited_pages = [int(p) for p in re.findall(r"[\[【]PAGE:(\d+)[\]】]", answer)]
    cited_pages = [int(p) for p in re.findall(r"[\[【]PAGE:(\d+)[\]】]", answer)]
    ungrounded = [p for p in cited_pages if p not in retrieved_pages]
    return {
        "cited_pages": cited_pages,
        "retrieved_pages": retrieved_pages,
        "ungrounded_citations": ungrounded,
        "fully_grounded": len(ungrounded) == 0,
    }

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="/data/ranjith/lang_graph/week2/chroma_papers_db",
    embedding_function=embeddings,
)

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
    system_msg = {
        "role": "system",
        "content": (
            "When citing a source, use EXACTLY this format: [PAGE:N] where N is the "
            "page number, e.g. [PAGE:6]. Do not use any other citation style. "
            "Only cite pages that appear in RETRIEVED PAGES."
        ),
    }
    messages = [system_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()

result = graph.invoke({"messages": [{"role": "user", "content": "According to my papers, what causes KV-cache divergence?"}]})

final_answer = result["messages"][-1].content
print(final_answer)
print(vectorstore._collection.count())

# pull RETRIEVED PAGES back out of the tool message
retrieved_pages = []
for msg in result["messages"]:
    if msg.type == "tool":
        match = re.search(r"RETRIEVED PAGES: \[([\d, ]+)\]", msg.content)
        if match:
            retrieved_pages = [int(p.strip()) for p in match.group(1).split(",")]

report = check_citation_grounding(final_answer, retrieved_pages)
print("\n=== Grounding check ===")
print(report)