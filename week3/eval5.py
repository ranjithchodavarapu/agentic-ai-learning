from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from collections import Counter

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="/data/ranjith/lang_graph/week2/chroma_papers_db", embedding_function=embeddings)

@tool
def search_papers(query: str) -> str:
    """Search Ranjith's reference papers about KV-cache divergence and decoding reliability."""
    results = vectorstore.similarity_search(query, k=2)
    if not results:
        return "NO RESULTS FOUND."
    return "\n\n---\n\n".join(f"[page {d.metadata.get('page')}]\n{d.page_content}" for d in results)

llm = ChatOllama(model="llama3.1:8b", temperature=0.0)

# each question tested both ungrounded and grounded, same design as Day 4, generalized
test_questions = [
    "What causes KV-cache divergence?",
    "How does FP32 affect divergence?",
    "What role does GQA play in divergence amplification?",
    "What is greedy decoding failure?",
]

N_TRIALS = 10  # lower than Day 4's 30, since this now runs 4 questions x 2 conditions x 10 = 80 calls

def consistency_rate(outputs):
    counts = Counter(outputs)
    return counts.most_common(1)[0][1] / len(outputs)

def run_stability_test(prompt, n_trials=N_TRIALS):
    outputs = []
    for i in range(n_trials):
        response = llm.invoke(prompt)
        outputs.append(response.content.strip())
    return outputs

def build_grounded_prompt(question: str) -> tuple[str, str]:
    """Retrieves once and builds a grounded prompt. Returns (prompt, retrieved_context)
    so you can inspect what was actually retrieved for each question."""
    context = search_papers.invoke({"query": question})
    prompt = f"""Answer the question using ONLY the information below. Do not use outside knowledge.

{context}

Question: {question}"""
    return prompt, context

results_table = []

for question in test_questions:
    print(f"\n=== {question} ===", flush=True)

    print("  ungrounded...", flush=True)
    ungrounded_outputs = run_stability_test(question)
    ungrounded_rate = consistency_rate(ungrounded_outputs)

    print("  grounded...", flush=True)
    grounded_prompt, retrieved = build_grounded_prompt(question)
    grounded_outputs = run_stability_test(grounded_prompt)
    grounded_rate = consistency_rate(grounded_outputs)

    print(f"  ungrounded consistency: {ungrounded_rate:.0%}", flush=True)
    print(f"  grounded consistency:   {grounded_rate:.0%}", flush=True)

    results_table.append({
        "question": question,
        "ungrounded_consistency": ungrounded_rate,
        "grounded_consistency": grounded_rate,
        "retrieval_found_content": retrieved != "NO RESULTS FOUND.",
    })

print("\n\n=== SUMMARY TABLE ===")
print(f"{'Question':<45} {'Ungrounded':<12} {'Grounded':<12} {'Delta':<8}")
for r in results_table:
    delta = r["grounded_consistency"] - r["ungrounded_consistency"]
    print(f"{r['question'][:44]:<45} {r['ungrounded_consistency']:<12.0%} {r['grounded_consistency']:<12.0%} {delta:+.0%}")

for r in results_table:
    print(r["question"], "-> retrieval found content:", r["retrieval_found_content"])

context = search_papers.invoke({"query": "What is greedy decoding failure?"})
print(context[:500])

