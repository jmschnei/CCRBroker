"""Tests para routedata.agents: factory y comportamiento de los 6 agentes de baseline.

Sin datos externos, sin red, sin GPU.
Ejecuta:  pytest tests/test_agents.py
"""
import pytest

from routedata.agents import RoutingAgent, RoutingDecision, Candidate, create_agent
from routedata.agents.baseline import (
    CostAgent,
    HeuristicAgent,
    HybridAgent,
    LatencyAgent,
    QualityAgent,
    RandomAgent,
)


def _candidates():
    return [
        Candidate(name="cheap", cost_per_1k_tokens=0.5, quality_score=0.6, latency_ms=200),
        Candidate(name="mid", cost_per_1k_tokens=2.0, quality_score=0.8, latency_ms=600),
        Candidate(name="top", cost_per_1k_tokens=8.0, quality_score=0.95, latency_ms=2000),
    ]


def test_factory_returns_known_agents():
    names = {"random", "cost", "quality", "latency", "hybrid", "heuristic"}
    for name in names:
        agent = create_agent(name)
        assert isinstance(agent, RoutingAgent)
        decision = agent.decide("What is the capital of France?", _candidates())
        assert isinstance(decision, RoutingDecision)
        assert decision.chosen_model in {"cheap", "mid", "top"}
        assert decision.query == "What is the capital of France?"
        assert decision.reasoning  # toda decisión debe ser explicable


def test_factory_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown agent"):
        create_agent("does-not-exist")


def test_cost_agent_picks_cheapest():
    decision = CostAgent().decide("q", _candidates())
    assert decision.chosen_model == "cheap"
    assert decision.all_scores["cheap"] > decision.all_scores["top"]  # score = -coste


def test_quality_agent_picks_best_quality():
    decision = QualityAgent().decide("q", _candidates())
    assert decision.chosen_model == "top"
    assert decision.all_scores["top"] == pytest.approx(0.95)


def test_latency_agent_picks_fastest():
    decision = LatencyAgent().decide("q", _candidates())
    assert decision.chosen_model == "cheap"


def test_hybrid_weights_are_configurable():
    quality_only = create_agent("hybrid", {"w_quality": 1.0, "w_cost": 0.0, "w_latency": 0.0})
    assert isinstance(quality_only, HybridAgent)
    assert quality_only.decide("q", _candidates()).chosen_model == "top"

    cost_only = create_agent("hybrid", {"w_quality": 0.0, "w_cost": 1.0, "w_latency": 0.0})
    assert cost_only.decide("q", _candidates()).chosen_model == "cheap"


def test_heuristic_detects_language_and_domain():
    agent = HeuristicAgent()
    assert agent._detect_language("¿Qué es la fotosíntesis y para qué sirve?") == "spanish"
    assert agent._detect_language("Der Vertrag ist sehr wichtig") == "german"
    assert agent._detect_domain("What does the contract say about liability in court?") == "legal"
    assert agent._detect_domain("Write code: implement a database query") == "technical"
    assert agent._detect_domain("I need a diagnosis for my patient, explain the treatment") == "medical"
    assert agent._detect_domain("What is 2 + 2?") == "general"


def test_heuristic_complexity_bonuses_strong_models():
    agent = HeuristicAgent()
    cands = [Candidate(name="weak", quality_score=0.6), Candidate(name="strong", quality_score=0.75)]
    short = agent.decide("hi", cands)
    assert short.all_scores["strong"] == pytest.approx(0.75)
    long_q = " ".join(["please explain in detail"] * 40)  # >200 words -> complexity 1.0
    long_d = agent.decide(long_q, cands)
    assert long_d.all_scores["strong"] == pytest.approx(0.85)  # +0.1 bonus por complejidad
    assert "complexity" in long_d.reasoning
    assert "language" in long_d.reasoning


def test_heuristic_prefers_higher_quality():
    decision = create_agent("heuristic").decide("Explain this", _candidates())
    assert decision.chosen_model == "top"


def test_random_agent_is_seeded_and_deterministic():
    cands = _candidates()
    first = RandomAgent(seed=7).decide("q", cands)
    second = RandomAgent(seed=7).decide("q", cands)
    assert first.chosen_model == second.chosen_model


def test_empty_candidates_raise():
    with pytest.raises(ValueError, match="No candidates"):
        RandomAgent().decide("q", [])


def test_missing_optional_fields_do_not_crash():
    cands = [Candidate(name="unknown"), Candidate(name="costly", cost_per_1k_tokens=9.0)]
    for name in ("cost", "quality", "latency", "hybrid", "heuristic"):
        decision = create_agent(name).decide("q", cands)
        assert decision.chosen_model in {"unknown", "costly"}
