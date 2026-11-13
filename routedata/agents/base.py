"""Base classes for routing agents.

A routing agent takes a query (and optional candidate models) and returns
a routing decision: which model to use and why.

The agent does NOT execute the query — it only decides the route.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RoutingStrategy(str, Enum):
    """Strategy used to make the routing decision."""

    COST = "cost"
    QUALITY = "quality"
    LATENCY = "latency"
    HEURISTIC = "heuristic"
    LEARNED = "learned"


@dataclass
class Candidate:
    """A candidate model that can handle the query.

    Attributes:
        name: Unique model name (e.g., "gpt-4o", "mistral-7b").
        capabilities: List of capabilities (e.g., ["text", "vision"]).
        cost_per_1k_tokens: Cost in EUR per 1k tokens (input + output average).
        quality_score: Normalized quality score [0, 1] if known, else None.
        latency_ms: Average latency in milliseconds if known, else None.
        context_window: Max context window in tokens if known, else None.
        metadata: Additional metadata.
    """

    name: str
    capabilities: list[str] = field(default_factory=list)
    cost_per_1k_tokens: float | None = None
    quality_score: float | None = None
    latency_ms: float | None = None
    context_window: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """The output of a routing decision.

    Attributes:
        query: The original query.
        chosen_model: Name of the model to use.
        strategy: Which strategy was used.
        confidence: Confidence score [0, 1] if available.
        reasoning: Human-readable explanation of the decision.
        all_scores: Scores for all candidates (model name -> score).
        latency_ms: How long the decision took.
    """

    query: str
    chosen_model: str
    strategy: RoutingStrategy
    confidence: float | None = None
    reasoning: str = ""
    all_scores: dict[str, float] = field(default_factory=dict)
    latency_ms: float | None = None


class RoutingAgent:
    """Base class for all routing agents.

    Subclasses must implement:
        - decide(query, candidates) -> RoutingDecision

    The agent should:
        1. Analyze the query (language, domain, complexity, etc.)
        2. Score each candidate model
        3. Select the best model
        4. Return a RoutingDecision with explanation
    """

    name: str = "base"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def decide(self, query: str, candidates: list[Candidate]) -> RoutingDecision:
        """Make a routing decision for the given query.

        Args:
            query: The user's query/question.
            candidates: List of candidate models to choose from.

        Returns:
            A RoutingDecision with the chosen model and reasoning.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError

    def _log(self, message: str) -> None:
        """Log a message (override for custom logging)."""
        pass
