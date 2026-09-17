import re

def categorize_answer(answer: str) -> str:
    if re.match(r'^\s*\{[\s\S]*"(type|name|function|parameters)"', answer):
        return "json_leak"
    declined = any(phrase in answer.lower() for phrase in [
        "don't have", "cannot determine", "not available", "no information",
        "unable to", "doesn't specify", "not specified", "don't know",
        "not explicitly mentioned", "not possible to", "not directly stated",
    ])
    return "declined" if declined else "answered"

# paste your saved eval2_overnight.log results here as a list of (question, answer_preview) tuples
# or just run this over the raw log file directly — see below
