"""Baseline routing agents for testing and comparison.

These agents use simple, explainable strategies:

1. **RandomAgent**: Randomly picks a model (baseline).
2. **CostAgent**: Always picks the cheapest model.
3. **QualityAgent**: Always picks the highest-quality model.
4. **LatencyAgent**: Always picks the lowest-latency model.
5. **HybridAgent**: Combines cost + quality + latency with weights.
6. **HeuristicAgent**: Uses simple heuristics based on query features
   (length, domain, language, complexity).
"""

from __future__ import annotations

import random
from typing import Any

from .base import Candidate, RoutingAgent, RoutingDecision, RoutingStrategy


class RandomAgent(RoutingAgent):
    """Baseline: randomly picks a model."""

    name = "random"

    def __init__(self, config: dict[str, Any] | None = None, seed: int = 42):
        super().__init__(config)
        self._rng = random.Random(seed)

    def decide(self, query: str, candidates: list[Candidate]) -> RoutingDecision:
        if not candidates:
            raise ValueError("No candidates provided")
        chosen = self._rng.choice(candidates)
        return RoutingDecision(
            query=query,
            chosen_model=chosen.name,
            strategy=RoutingStrategy.QUALITY,  # placeholder
            reasoning=f"Random selection: {chosen.name}",
        )


class CostAgent(RoutingAgent):
    """Baseline: always picks the cheapest model."""

    name = "cost"

    def decide(self, query: str, candidates: list[Candidate]) -> RoutingDecision:
        scored = []
        for c in candidates:
            cost = c.cost_per_1k_tokens if c.cost_per_1k_tokens is not None else 1.0
            # Lower cost is better, so we invert
            score = -cost
            scored.append((c.name, score))
        scored.sort(key=lambda x: x[1], reverse=True)

        chosen_name = scored[0][0]
        chosen_model = next(c for c in candidates if c.name == chosen_name)

        return RoutingDecision(
            query=query,
            chosen_model=chosen_name,
            strategy=RoutingStrategy.COST,
            reasoning=f"Cheapest model: {chosen_name} (cost={chosen_model.cost_per_1k_tokens} EUR/1k)",
            all_scores={name: score for name, score in scored},
        )


class QualityAgent(RoutingAgent):
    """Baseline: always picks the highest-quality model."""

    name = "quality"

    def decide(self, query: str, candidates: list[Candidate]) -> RoutingDecision:
        scored = []
        for c in candidates:
            quality = c.quality_score if c.quality_score is not None else 0.5
            scored.append((c.name, quality))
        scored.sort(key=lambda x: x[1], reverse=True)

        chosen_name = scored[0][0]

        return RoutingDecision(
            query=query,
            chosen_model=chosen_name,
            strategy=RoutingStrategy.QUALITY,
            reasoning=f"Highest quality model: {chosen_name}",
            all_scores={name: score for name, score in scored},
        )


class LatencyAgent(RoutingAgent):
    """Baseline: always picks the lowest-latency model."""

    name = "latency"

    def decide(self, query: str, candidates: list[Candidate]) -> RoutingDecision:
        scored = []
        for c in candidates:
            latency = c.latency_ms if c.latency_ms is not None else 1000.0
            # Lower latency is better
            score = -latency
            scored.append((c.name, score))
        scored.sort(key=lambda x: x[1], reverse=True)

        chosen_name = scored[0][0]

        return RoutingDecision(
            query=query,
            chosen_model=chosen_name,
            strategy=RoutingStrategy.LATENCY,
            reasoning=f"Lowest latency model: {chosen_name}",
            all_scores={name: score for name, score in scored},
        )


class HybridAgent(RoutingAgent):
    """Combines cost, quality, and latency with configurable weights.

    Score = w_quality * quality - w_cost * cost - w_latency * latency
    """

    name = "hybrid"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.w_quality = self.config.get("w_quality", 0.5)
        self.w_cost = self.config.get("w_cost", 0.3)
        self.w_latency = self.config.get("w_latency", 0.2)

    def decide(self, query: str, candidates: list[Candidate]) -> RoutingDecision:
        scored = []
        for c in candidates:
            quality = c.quality_score if c.quality_score is not None else 0.5
            cost = c.cost_per_1k_tokens if c.cost_per_1k_tokens is not None else 1.0
            latency = c.latency_ms if c.latency_ms is not None else 1000.0

            # Normalize cost and latency (assume max cost=10, max latency=5000ms)
            cost_norm = min(cost / 10.0, 1.0)
            latency_norm = min(latency / 5000.0, 1.0)

            score = self.w_quality * quality - self.w_cost * cost_norm - self.w_latency * latency_norm
            scored.append((c.name, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        chosen_name = scored[0][0]

        return RoutingDecision(
            query=query,
            chosen_model=chosen_name,
            strategy=RoutingStrategy.QUALITY,
            reasoning=(
                f"Hybrid score (w_q={self.w_quality}, w_c={self.w_cost}, w_l={self.w_latency}): {chosen_name}"
            ),
            all_scores={name: score for name, score in scored},
        )


class HeuristicAgent(RoutingAgent):
    """Uses simple heuristics based on query features.

    Features extracted:
    - Language (English, Spanish, German, etc.)
    - Domain (legal, medical, technical, general)
    - Complexity (number of words, presence of technical terms)
    - Query type (factual, analytical, creative)
    """

    name = "heuristic"

    # Domain-specific keywords
    DOMAIN_KEYWORDS = {
        "legal": [
            "law",
            "legal",
            "contract",
            "court",
            "regulation",
            "statute",
            "jurisprudence",
            "recht",
            "gesetz",
            "vertrags",
        ],
        "medical": [
            "medical",
            "patient",
            "diagnosis",
            "symptom",
            "treatment",
            "disease",
            "symptome",
            "behandlung",
        ],
        "technical": [
            "algorithm",
            "code",
            "software",
            "api",
            "database",
            "programming",
            "implement",
            "algorithmus",
            "code",
        ],
        "financial": ["finance", "investment", "stock", "market", "economy", "finanzen", "investment"],
        "general": [],
    }

    def _detect_language(self, text: str) -> str:
        """Simple language detection based on common words."""
        text_lower = text.lower()
        if any(w in text_lower for w in ["el", "la", "los", "las", "es", "y", "pero", "cómo"]):
            return "spanish"
        if any(w in text_lower for w in ["der", "die", "das", "ist", "und", "aber"]):
            return "german"
        return "english"

    def _detect_domain(self, text: str) -> str:
        """Detect domain based on keywords."""
        text_lower = text.lower()
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            if domain == "general":
                continue
            if any(kw in text_lower for kw in keywords):
                return domain
        return "general"

    def _estimate_complexity(self, text: str) -> float:
        """Estimate complexity based on word count and technical terms."""
        words = len(text.split())
        # Simple heuristic: more words = more complex
        if words < 10:
            return 0.2
        elif words < 50:
            return 0.5
        elif words < 200:
            return 0.8
        else:
            return 1.0

    def _select_model_by_domain(self, domain: str, candidates: list[Candidate]) -> Candidate | None:
        """Select a model based on domain (if candidates have domain-specific scores)."""
        # For now, just return None (use default scoring)
        # In the future, we can add domain-specific model preferences
        return None

    def decide(self, query: str, candidates: list[Candidate]) -> RoutingDecision:
        # Extract features
        language = self._detect_language(query)
        domain = self._detect_domain(query)
        complexity = self._estimate_complexity(query)

        # Score each candidate
        scored = []
        for c in candidates:
            base_score = 0.5  # Default

            # If we have quality scores, use them
            if c.quality_score is not None:
                base_score = c.quality_score

            # Complexity bonus for stronger models
            if complexity > 0.7 and c.quality_score is not None and c.quality_score > 0.7:
                base_score += 0.1  # Bonus for complex queries

            scored.append((c.name, base_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        chosen_name = scored[0][0]

        return RoutingDecision(
            query=query,
            chosen_model=chosen_name,
            strategy=RoutingStrategy.HEURISTIC,
            reasoning=(
                f"Heuristic: language={language}, domain={domain}, complexity={complexity:.2f}. "
                f"Chose: {chosen_name}"
            ),
            all_scores={name: score for name, score in scored},
        )


# Convenience factory
def create_agent(name: str, config: dict[str, Any] | None = None) -> RoutingAgent:
    """Create a routing agent by name."""
    agents: dict[str, type[RoutingAgent]] = {
        "random": RandomAgent,
        "cost": CostAgent,
        "quality": QualityAgent,
        "latency": LatencyAgent,
        "hybrid": HybridAgent,
        "heuristic": HeuristicAgent,
    }
    if name not in agents:
        raise ValueError(f"Unknown agent: {name}. Available: {list(agents.keys())}")
    return agents[name](config)
