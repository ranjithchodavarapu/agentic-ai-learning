# Agentic AI → RAG → Decoding-Reliability Eval Harness

A month-long, self-directed project: starting from LangGraph fundamentals,
building up through RAG and multi-agent architectures, to a controlled
experiment testing whether retrieval grounding affects LLM decoding
stability — not just answer correctness. Built and evaluated on local
inference (Ollama, llama3.1:8b) on a university H100 cluster, after
exhausting three separate free-tier API providers along the way.

## The core finding

**Retrieval grounding appears to stabilize decoding itself, not just
correct the answer's content.** A controlled comparison (n=30 trials/
condition, temperature=0.0, identical retrieved context reused across all
grounded trials to isolate decoding variance from retrieval variance):

| Condition | Consistency | Topical accuracy |
|---|---|---|
| Ungrounded | 97% | 0/30 — consistently misinterpreted "KV-cache divergence" as a distributed-systems caching concept |
| Grounded | 100% | 30/30 — correctly reproduced the paper's actual finding (FP16 format, not the cache mechanism, as the cause) |

The pattern generalized across 3/4 further test questions (the fourth was
already at a 100% ungrounded ceiling, so no effect was observable) and held
up under a negative control: a genuinely off-corpus question ("capital of
Australia") returned real-but-irrelevant retrieved content, and the model
correctly declined to force a connection in 10/10 trials rather than
fabricating one or silently ignoring the grounding instruction.

Try it live: `week4/app.py` (Streamlit demo, runs both conditions on any
question against the paper corpus in real time).

## Structure

- **`week1/`** — LangGraph fundamentals: state/nodes/edges, conditional
  routing, checkpointed memory, MCP tool integration (a local `add` tool
  exposed via a real MCP server, mixed with a local tool in one agent).
- **`week2/`** — RAG: chunking, local embeddings, a grounded Q&A agent over
  my own reference papers, with automated citation-grounding checks.
  Finding: an explicit output-format instruction (`[PAGE:N]` citations) was
  followed in only ~50% of identical calls, despite content accuracy
  holding at 100% — a measurable gap between correctness and instruction
  compliance.
- **`week3/`** — Multi-agent supervisor pattern, then the main eval harness:
  the grounding-stability experiment above. Also: a capability-boundary
  fabrication finding (an agent invented specific, inconsistent page counts
  for a question its tool couldn't structurally answer) and a tool-calling
  protocol failure specific to local 8B inference (13% of trials leaked raw
  JSON instead of a real answer).
- **`week4/`** — Packaged the core finding as an interactive Streamlit demo.

Each week's folder has its own `readme.md` with full methodology and
additional findings not summarized here.

## A recurring methodological theme

Across all four weeks, roughly five separate times, a hand-written
automated check (a citation-format regex, a relevance-score threshold, a
decline-detection keyword list, a shell `grep` pattern) silently reported a
misleading result — a false pass, an undercounted rate, or a nonsense
number — instead of erroring. Every real finding in this project only
became trustworthy after reading raw model output directly rather than
trusting an aggregate metric. That discipline — treat a clean-looking
number as a hypothesis to check, not a conclusion — is the same standard my
actual PhD research on LLM inference reliability already requires; this
project was a chance to apply it to a different kind of system (agents and
RAG rather than raw decoding), and it held up.

## Setup

```bash
pip install -r requirements.txt
ollama pull llama3.1:8b   # requires Ollama installed; see week1/readme.md
```
Each week's scripts assume the vectorstore built in `week2/` and the local
Ollama server running (`ollama serve`).
