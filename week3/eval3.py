from langchain_ollama import ChatOllama
import re

test_prompts = [
    "What is the capital of France?",
    "What causes KV-cache divergence?",
    "List three prime numbers greater than 20.",
]

temperatures = [0.0, 0.3, 0.7, 1.0]
N_TRIALS = 10

def run_temperature_sweep(prompts, temps, n_trials=N_TRIALS):
    results = []
    for prompt in prompts:
        for temp in temps:
            llm = ChatOllama(model="llama3.1:8b", temperature=temp)
            outputs = []
            for trial in range(n_trials):
                response = llm.invoke(prompt)
                outputs.append(response.content.strip())
            results.append({
                "prompt": prompt[:40],
                "temperature": temp,
                "outputs": outputs,
            })
            print(f"Done: {prompt[:30]} @ temp={temp}", flush=True)
    return results

sweep_results = run_temperature_sweep(test_prompts, temperatures)

def consistency_rate(outputs: list[str]) -> float:
    """Fraction of trials that exactly match the most common output."""
    from collections import Counter
    counts = Counter(outputs)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(outputs)

for r in sweep_results:
    rate = consistency_rate(r["outputs"])
    print(f"{r['prompt']} @ temp={r['temperature']}: consistency={rate:.0%}")

for r in sweep_results:
    if "capital" in r["prompt"] and r["temperature"] == 1.0:
        for o in r["outputs"]:
            print(repr(o))

for r in sweep_results:
    if "KV-cache" in r["prompt"] and r["temperature"] == 0.0:
        from collections import Counter
        counts = Counter(r["outputs"])
        for output, count in counts.items():
            print(f"=== ({count}x) ===")
            print(output)
            print()