from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.tools import tool
import re

def check_citation_grounding(answer: str, retrieved_pages: list[int]) -> dict:
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

# --- CALIBRATION STEP — run this once, read the numbers, then set RELEVANCE_THRESHOLD below ---
print("=== Calibrating distance threshold ===")
for q in ["what causes KV-cache divergence?", "airspeed velocity of an unladen swallow"]:
    print(f"\n--- {q} ---")
    for doc, score in vectorstore.similarity_search_with_score(q, k=3):
        print(f"  distance={score:.3f}  page={doc.metadata.get('page')}")
print("=== end calibration — set RELEVANCE_THRESHOLD between the two clusters above ===\n")
# --- end calibration block — delete once you've picked a real threshold ---

RELEVANCE_THRESHOLD = 1.0  # PLACEHOLDER — replace with a real value from the calibration output above

@tool
def search_papers(query: str) -> str:
    """Search Ranjith's reference papers for passages about KV-cache divergence,
    decoding reliability, or FP16 non-associativity research. Only cite the exact
    source and page numbers shown below — never cite a page that does not appear here."""
    results_with_scores = vectorstore.similarity_search_with_score(query, k=3)
    relevant = [(doc, score) for doc, score in results_with_scores if score < RELEVANCE_THRESHOLD]
    if not relevant:
        return "NO RELEVANT RESULTS FOUND. Tell the user your papers don't cover this topic."
    retrieved_pages = [doc.metadata.get("page") for doc, score in relevant]
    formatted = "\n\n---\n\n".join(
        f"[{doc.metadata.get('source')}, page {doc.metadata.get('page')}] (distance: {score:.3f})\n{doc.page_content}"
        for doc, score in relevant
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

eval_questions = [
    {"question": "According to my papers, what causes KV-cache divergence?", "answerable": True},
    {"question": "How does the GQA ratio relate to divergence amplification?", "answerable": True},
    {"question": "What happens to divergence when running in FP32 instead of FP16?", "answerable": True},
    {"question": "According to my papers, what is the airspeed velocity of an unladen swallow?", "answerable": False},
    {"question": "According to my papers, how does quantum entanglement affect transformer attention?", "answerable": False},
]

def run_eval(questions, n_trials=2):
    all_results = []
    for q in questions:
        for trial in range(n_trials):
            try:
                result = graph.invoke({"messages": [{"role": "user", "content": q["question"]}]})
                final_answer = result["messages"][-1].content
            except Exception as e:
                all_results.append({
                    "question": q["question"][:50],
                    "trial": trial,
                    "expected_answerable": q["answerable"],
                    "error": str(e)[:200],
                    "fully_grounded": None,
                    "cited_pages": None,
                    "used_ascii_brackets": None,
                    "correctly_declined_if_unanswerable": None,
                })
                continue

            retrieved_pages = []
            for msg in result["messages"]:
                if msg.type == "tool":
                    match = re.search(r"RETRIEVED PAGES: \[([\d, ]+)\]", msg.content)
                    if match:
                        retrieved_pages = [int(p.strip()) for p in match.group(1).split(",")]

            grounding = check_citation_grounding(final_answer, retrieved_pages)
            correctly_declined = len(retrieved_pages) == 0  # no relevant chunks retrieved = correct grounds for declining

            all_results.append({
                "question": q["question"][:50],
                "trial": trial,
                "expected_answerable": q["answerable"],
                "error": None,
                "fully_grounded": grounding["fully_grounded"],
                "cited_pages": grounding["cited_pages"],
                "retrieved_pages": retrieved_pages,   # ← new line, for visibility
                "used_ascii_brackets": bool(re.search(r"\[PAGE:\d+\]", final_answer)),
                "correctly_declined_if_unanswerable": correctly_declined if not q["answerable"] else None,
                "answer_preview": final_answer[:150],
            })
    return all_results
results = run_eval(eval_questions, n_trials=2)
for r in results:
    print(r)

valid_results = [r for r in results if r["error"] is None]
answerable_results = [r for r in valid_results if r["expected_answerable"]]
unanswerable_results = [r for r in valid_results if not r["expected_answerable"]]

grounding_rate = sum(r["fully_grounded"] for r in answerable_results) / len(answerable_results) if answerable_results else 0
ascii_rate = sum(r["used_ascii_brackets"] for r in answerable_results) / len(answerable_results) if answerable_results else 0
correct_decline_rate = sum(r["correctly_declined_if_unanswerable"] for r in unanswerable_results) / len(unanswerable_results) if unanswerable_results else 0

print(f"\nGrounding rate (answerable questions): {grounding_rate:.0%}")
print(f"ASCII format compliance rate: {ascii_rate:.0%}")
print(f"Correct-decline rate (unanswerable questions): {correct_decline_rate:.0%}")
print(f"Errors: {len(results) - len(valid_results)}/{len(results)}")