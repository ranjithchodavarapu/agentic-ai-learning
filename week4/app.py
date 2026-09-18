import streamlit as st
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from collections import Counter
import requests 

st.set_page_config(page_title="Grounding Stability Demo", layout="wide")

try:
    requests.get("http://localhost:11434", timeout=2)
    st.success("Ollama server: connected", icon="✅")
except Exception:
    st.error("Ollama server not reachable — check `tmux attach -t ollama`", icon="⚠️")


@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="/data/ranjith/lang_graph/week2/chroma_papers_db", embedding_function=embeddings)
    llm = ChatOllama(model="llama3.1:8b", temperature=0.0)
    return vectorstore, llm

vectorstore, llm = load_resources()

@tool
def search_papers(query: str) -> str:
    """Search Ranjith's reference papers about KV-cache divergence and decoding reliability."""
    results = vectorstore.similarity_search(query, k=2)
    if not results:
        return "NO RESULTS FOUND."
    return "\n\n---\n\n".join(f"[page {d.metadata.get('page')}]\n{d.page_content}" for d in results)

def consistency_rate(outputs):
    counts = Counter(outputs)
    return counts.most_common(1)[0][1] / len(outputs), counts

st.title("Does Grounding Stabilize Decoding?")
st.markdown(
    "A live demo of the Week 3 finding: retrieval grounding doesn't just improve "
    "*correctness* — it appears to reduce decoding *non-determinism* itself, even at "
    "temperature=0.0. Enter a question about KV-cache divergence / decoding reliability "
    "research and run both conditions."
)

question = st.text_input("Question", value="What causes KV-cache divergence?")
n_trials = st.slider("Trials per condition", 3, 20, 10)

if st.button("Run comparison"):
    with st.spinner(f"Running {n_trials} ungrounded trials..."):
        ungrounded_outputs = [llm.invoke(question).content.strip() for _ in range(n_trials)]
        ungrounded_rate, ungrounded_counts = consistency_rate(ungrounded_outputs)

    context = search_papers.invoke({"query": question})
    grounded_prompt = f"""Answer the question using ONLY the information below. Do not use outside knowledge.

{context}

Question: {question}"""

    with st.spinner(f"Running {n_trials} grounded trials..."):
        grounded_outputs = [llm.invoke(grounded_prompt).content.strip() for _ in range(n_trials)]
        grounded_rate, grounded_counts = consistency_rate(grounded_outputs)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Ungrounded consistency", f"{ungrounded_rate:.0%}")
        st.text_area("Most common ungrounded answer", ungrounded_counts.most_common(1)[0][0], height=200)
    with col2:
        st.metric("Grounded consistency", f"{grounded_rate:.0%}", delta=f"{(grounded_rate-ungrounded_rate):+.0%}")
        st.text_area("Most common grounded answer", grounded_counts.most_common(1)[0][0], height=200)

    with st.expander("Retrieved context used for grounding"):
        st.text(context)

    with st.expander(f"All {n_trials} ungrounded outputs (distinct variants)"):
        for o, c in ungrounded_counts.most_common():
            st.markdown(f"**({c}x)** {o}")

    with st.expander(f"All {n_trials} grounded outputs (distinct variants)"):
        for o, c in grounded_counts.most_common():
            st.markdown(f"**({c}x)** {o}")