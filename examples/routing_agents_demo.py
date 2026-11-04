"""Demo: routea una query entre varios modelos de pesos abiertos con los agentes de baseline.

No requiere datos, red ni GPU.
Uso:  python examples/routing_agents_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from routedata.agents import Candidate, create_agent

# Pool hipotético de modelos autoalojados (precios relativos)
CANDIDATES = [
    Candidate(name="qwen2.5-7b-instruct", cost_per_1k_tokens=0.05, quality_score=0.62, latency_ms=180),
    Candidate(name="llama-3.1-8b-instruct", cost_per_1k_tokens=0.08, quality_score=0.68, latency_ms=240),
    Candidate(name="deepseek-r1-distill-14b", cost_per_1k_tokens=0.15, quality_score=0.81, latency_ms=600),
    Candidate(name="qwen3-30b-a3b-instruct", cost_per_1k_tokens=0.30, quality_score=0.88, latency_ms=900),
]

QUERY = "Explain, step by step, how rotary position embeddings work in transformer attention."


def main() -> None:
    print(f"Query: {QUERY}\n")
    print(f"{'agente':<10} {'modelo elegido':<28} {'score':>8}  razon")
    print("-" * 100)
    for name in ("cost", "quality", "latency", "hybrid", "heuristic"):
        decision = create_agent(name).decide(QUERY, CANDIDATES)
        score = decision.all_scores.get(decision.chosen_model, float("nan"))
        print(f"{name:<10} {decision.chosen_model:<28} {score:>8.3f}  {decision.reasoning}")


if __name__ == "__main__":
    main()
