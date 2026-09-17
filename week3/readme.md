# Week 3 — Multi-agent patterns + decoding-sensitivity eval harness

A LangGraph supervisor architecture (Days 1-2) and a controlled decoding-stability
harness (Days 3-5) built on local inference (Ollama, llama3.1:8b) after exhausting
three separate free-tier API providers in one debugging session.

## Day 1-2: Multi-agent supervisor pattern
- Extended `MessagesState` with a routing field; a supervisor node uses
  `with_structured_output` (Pydantic schema) to route to a `research_agent`
  (bound to `search_papers`) or `math_agent` (bound to `add`), each with its
  own isolated `ToolNode`.
- **Finding — capability-boundary fabrication**: asked "How many pages is my
  KV-cache paper?" — a question `search_papers` cannot structurally answer
  (it returns content chunks, not document metadata) — the agent fabricated
  a confident, specific page count in every trial where it didn't error,
  with values disagreeing across repeated calls (13, 13, 13, 2, 16, "11 and
  13"...). This is distinct from Week 2's "no relevant content" case, where
  the model correctly declined: here the tool *was* relevant enough to be
  called, but structurally incapable of answering, and the model didn't
  recognize that gap.
- **Infrastructure finding**: multi-agent graphs cost 2-3x the API calls per
  test case vs. a single agent (supervisor call + routed agent call + tool
  loop), which matters a lot for free-tier eval budgets. Confirmed the hard
  way — exhausted Groq's 200K token/day cap on two different models
  (`gpt-oss-120b`, `gpt-oss-20b`) and hit Gemini's 20-requests/day free
  quota, all before a single 18-call eval sweep completed. Fixed by
  installing Ollama without sudo on the H100 (`/data/ranjith/ollama`,
  binary + models redirected off the home-directory quota).

## Day 2 (continued): overnight local eval, capability-boundary sweep
- 60 trials on `llama3.1:8b`, no rate limits.
- **JSON-leak finding**: 8/60 trials (13%) returned raw, unparsed tool-call
  JSON as the visible answer instead of a real response — concentrated
  heavily on one question (7/10 for "How does FP32 affect divergence?").
  This is a distinct failure category from hallucination: a tool-calling
  protocol breakdown specific to this local model via Ollama, not a
  content-accuracy problem.
- Confirmed fabrication at real sample size on tool-incapable questions
  (page count, publication year — no consistent correct value recovered
  across 30 trials).
- The keyword-based decline-detector undercounted real declines (missed
  phrasing like "not explicitly mentioned," "not possible to give an exact
  count") — the fifth instance this project of a hand-written text-matching
  check silently missing real signal, always caught by reading raw output
  rather than trusting the aggregate number.

## Day 3-5: Decoding-sensitivity / grounding-stability harness
Core question: does retrieval grounding affect not just answer *correctness*
but decoding *consistency* itself?

- **Day 3 (temperature sweep, n=10)**: closed-form factual questions ("capital
  of France") stay 100% consistent regardless of temperature — the answer's
  probability mass is too concentrated for sampling to matter. Open-ended
  questions ("What causes KV-cache divergence?") dropped from 90% consistency
  at temp=0.0 to ~10% at temp≥0.3, as expected. At temp=0.0 specifically,
  reading the full (not truncated) outputs showed the 1-in-10 outlier trial
  differed from the majority in real content (different number and set of
  claimed causes), not just phrasing — genuine non-determinism in nominally
  greedy decoding.
- **Day 4 (controlled comparison, n=30/condition)**: same question, ungrounded
  vs. grounded (retrieval fixed once, reused across all 30 trials to isolate
  decoding variance from retrieval variance). Ungrounded: 97% consistent but
  **topically wrong in 30/30 trials** — "KV-cache" collided with the more
  common distributed-systems-caching meaning in training data, since no
  retrieval was available to disambiguate. Grounded: **100% consistent and
  correct in 30/30 trials**, reproducing the paper's actual finding (FP16
  format, not the cache mechanism, as the cause) verbatim every time.
  Grounding eliminated both the topic-collision problem and the small
  residual decoding non-determinism.
- **Day 5 (generalization, n=10 x 4 questions)**: the pattern held for 3/4
  questions (+10 points consistency each, 90%→100%); the fourth was already
  at a 100% ungrounded ceiling, so no effect was observable. Confirmed via a
  `retrieval_found_content` check that all four questions retrieved genuinely
  relevant content — including a non-obvious connection between greedy
  decoding and FP16 accumulation order that the model correctly surfaced
  from the corpus.
- **Negative control**: a question entirely outside the corpus ("capital of
  Australia") returned real-but-irrelevant retrieved content (`search_papers`
  has no relevance threshold — always returns top-k regardless of match
  quality). The model correctly recognized the retrieved content as
  unrelated in 10/10 trials rather than forcing a spurious connection or
  silently reverting to outside knowledge — a distinct, positive
  instruction-following result from the grounding-comparison findings above.

## Files
- `eval1.py` — supervisor pattern, initial routing tests
- `eval2.py` — capability-boundary eval (page count / publication year /
  co-authors), overnight local run
- `eval3.py` — temperature sweep (France / KV-cache / primes)
- `eval4.py` — controlled grounded-vs-ungrounded comparison, n=30
- `eval5.py` — generalized grounded-vs-ungrounded across 4 questions
- `eval5b.py` — negative control (off-corpus question)

## What I learned
The most valuable result of the week wasn't any single number — it was
learning to distrust a clean-looking metric until I'd read the raw output
behind it. Five separate times this project, a hand-written check (citation
regex, relevance threshold, decline-detector keywords, a grep pattern, this
week's decline-detector again) silently reported a misleading result instead
of erroring. The actual finding I'm most confident in — that grounding
stabilizes decoding, not just corrects content — only became trustworthy once
I'd controlled for retrieval variance (Day 4), checked retrieval quality
directly rather than assuming it (`retrieval_found_content`), and read full
untruncated outputs instead of trusting a percentage. That discipline is the
same one my actual research already requires; this project was a chance to
apply it to a different kind of system.
