from pathlib import Path

import numpy as np
import yaml

from scripts.benchmark_vllm import _build_prompt
from scripts.run_phase1 import OPEN_WEIGHT_MODELS, _fit_predict

ROOT = Path(__file__).resolve().parents[1]


def test_all_pool_and_judge_revisions_are_pinned():
    pool = yaml.safe_load((ROOT / "configs" / "pool_openweights.yaml").read_text())
    entries = [*pool["models"].values(), *pool["judges"]]

    assert pool["revision_checked_at"] == "2026-11-13"
    assert len(pool["models"]) == 9
    assert len(pool["judges"]) == 3
    assert all(len(entry["revision"]) == 40 for entry in entries)
    assert all(set(entry["revision"]) <= set("0123456789abcdef") for entry in entries)


def test_build_prompt_respects_tokenizer_target():
    class WhitespaceTokenizer:
        @staticmethod
        def encode(text):
            return text.split()

        @staticmethod
        def decode(tokens):
            return " ".join(tokens)

    prompt = _build_prompt(WhitespaceTokenizer(), 20)

    assert len(WhitespaceTokenizer.encode(prompt)) == 20


def test_phase1_routers_return_only_valid_candidate_indices():
    train_texts = ["simple math question", "write a python function", "explain medical science", "short fact"]
    test_texts = ["solve a math problem", "implement code"]
    train_scores = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    candidates = list(OPEN_WEIGHT_MODELS[:3])

    for strategy in ("random", "best_single", "length_bucket", "tfidf_knn", "tfidf_classifier"):
        choices = _fit_predict(strategy, train_texts, test_texts, train_scores, candidates, seed=42)
        assert choices.shape == (len(test_texts),)
        assert ((choices >= 0) & (choices < len(candidates))).all()
