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

# Retrieve ONCE, outside the loop — this holds retrieval fixed across all trials,
# so any variation observed is isolated to decoding, not retrieval randomness.
retrieved_context = search_papers.invoke({"query": "What causes KV-cache divergence?"})

grounded_prompt = f"""Answer the question using ONLY the information below. Do not use outside knowledge.

{retrieved_context}

Question: What causes KV-cache divergence?"""

N_TRIALS = 30

def run_stability_test(prompt, n_trials=N_TRIALS):
    outputs = []
    for i in range(n_trials):
        response = llm.invoke(prompt)
        outputs.append(response.content.strip())
        print(f"  trial {i} done", flush=True)
    return outputs

print("=== Ungrounded, n=30, temp=0.0 ===", flush=True)
ungrounded_outputs = run_stability_test("What causes KV-cache divergence?")

print("=== Grounded (fixed retrieved context), n=30, temp=0.0 ===", flush=True)
grounded_outputs = run_stability_test(grounded_prompt)

def consistency_rate(outputs):
    counts = Counter(outputs)
    return counts.most_common(1)[0][1] / len(outputs)

print(f"\nUngrounded consistency: {consistency_rate(ungrounded_outputs):.0%}")
print(f"Grounded consistency: {consistency_rate(grounded_outputs):.0%}")

for name, outputs in [("Ungrounded", ungrounded_outputs), ("Grounded", grounded_outputs)]:
    print(f"\n--- {name}: distinct outputs, most common first ---")
    for output, count in Counter(outputs).most_common():
        print(f"({count}x) {output[:150]}")