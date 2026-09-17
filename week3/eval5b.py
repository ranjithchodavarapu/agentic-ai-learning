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

N_TRIALS = 10

def consistency_rate(outputs):
    counts = Counter(outputs)
    return counts.most_common(1)[0][1] / len(outputs)

def run_stability_test(prompt, n_trials=N_TRIALS):
    return [llm.invoke(prompt).content.strip() for _ in range(n_trials)]

question = "What is the capital of Australia?"

# check what retrieval actually returns for this off-topic question, before running trials
context = search_papers.invoke({"query": question})
print("=== Retrieved context for off-topic question ===")
print(context[:300])
print(f"\nretrieval_found_content: {context != 'NO RESULTS FOUND.'}")
print()

grounded_prompt = f"""Answer the question using ONLY the information below. Do not use outside knowledge.

{context}

Question: {question}"""

print(f"=== {question} ===")
print("  ungrounded...", flush=True)
ungrounded_outputs = run_stability_test(question)
ungrounded_rate = consistency_rate(ungrounded_outputs)

print("  grounded (with off-topic/empty context)...", flush=True)
grounded_outputs = run_stability_test(grounded_prompt)
grounded_rate = consistency_rate(grounded_outputs)

print(f"\n  ungrounded consistency: {ungrounded_rate:.0%}")
print(f"  grounded consistency:   {grounded_rate:.0%}")

print("\n--- Ungrounded distinct outputs ---")
for o, c in Counter(ungrounded_outputs).most_common():
    print(f"({c}x) {o[:150]}")

print("\n--- Grounded distinct outputs ---")
for o, c in Counter(grounded_outputs).most_common():
    print(f"({c}x) {o[:150]}")