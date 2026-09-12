# RAG mini-project — grounded Q&A over my own reference papers

A LangGraph agent (same architecture as Week 1) with a RAG tool: local embeddings
(sentence-transformers on the H100), Chroma vector store, and a calibrated
relevance threshold — evaluated systematically across 5 questions × 2 trials.

## What it demonstrates
- RAG-as-a-tool: retrieval is just another `@tool` in the same agent-loop
  architecture from Week 1 — no separate "RAG system," same `StateGraph`,
  same `tools_condition` loop.
- Free, local, GPU-backed embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
  instead of a hosted embedding API.
- Automated citation-grounding checks, forced output-format instructions, and
  a calibrated raw-distance relevance threshold (not LangChain's default
  `[0,1]` relevance-score normalization, which silently broke for this
  embedding model).

## Results (10 runs: 5 questions × 2 trials each)
- **Grounding rate: 100%** — every citation on answerable questions traced to
  a genuinely retrieved page.
- **Correct-decline rate: 100%** — out-of-scope questions (tested with
  deliberately absurd examples) were correctly refused once retrieval used a
  calibrated distance threshold instead of an uncalibrated default.
- **Citation format compliance: 67%** — an explicit, unambiguous instruction
  to use one citation format (`[PAGE:N]`) was followed in only 2/3 of
  identical calls; the model substituted full-width Unicode brackets
  (`【PAGE:N】`) the rest of the time. Content correctness and exact format
  compliance are measurably different reliability properties.

## The recurring pattern worth remembering
Three separate automated checks (citation-format regex, relevance-score
threshold, decline-detection keywords) each independently failed silently
before being fixed — in every case, a hardcoded pattern didn't match real
model output and returned a misleadingly clean or wrong result instead of
erroring. Lesson: prefer structural/numeric ground-truth signals (e.g. "was
anything actually retrieved") over parsing free-form model text, whenever a
ground-truth signal is available.

## Files
- `rag2.py` — PDF loading, chunking, embedding, persisted Chroma index
- `rag3.py` — RAG wrapped as a LangGraph tool, first grounding fixes
- `rag4.py` — automated citation-grounding checker, forced output format
- `rag5.py` — full eval sweep across multiple questions/trials, calibrated
  relevance threshold, final results above

## What I learned
The citation-format finding was the most interesting result of the week —
not because the model failed, but because it succeeded at content (100%,
every run) while failing at exact formatting instructions (67%) in a way
that was invisible unless I tested repeatably instead of trusting one clean
run. That distinction — and the fact that my own evaluation checks failed
silently multiple times before I caught it — feels directly relevant to the
kind of reliability questions I already work on, just applied to a new axis
(instruction-following under explicit constraint) instead of decoding
determinism.
