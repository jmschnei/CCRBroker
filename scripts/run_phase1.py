"""Run the five-seed LLMRouterBench Phase 1 model-selection experiment.

This implementation deliberately uses only ArenaHard's per-instance labels and
an explicit 20-model open-weight candidate list. It does not treat aggregate
benchmark scores as per-query labels and does not report costs/latency where the
source data has no meaningful cost variation or timing observations.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from routedata.splits import SEEDS_LLMROUTERBENCH, iid_split

DATASET = "arenahard"
OPEN_WEIGHT_MODELS = (
    "DeepHermes-3-Llama-3-8B-Preview",
    "DeepSeek-R1-0528-Qwen3-8B",
    "DeepSeek-R1-Distill-Qwen-7B",
    "Fin-R1",
    "GLM-Z1-9B-0414",
    "Intern-S1-mini",
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-8B-UltraMedical",
    "Llama-3.1-Nemotron-Nano-8B-v1",
    "MiMo-7B-RL-0530",
    "MiniCPM4.1-8B",
    "NVIDIA-Nemotron-Nano-9B-v2",
    "OpenThinker3-7B",
    "Qwen2.5-Coder-7B-Instruct",
    "Qwen3-8B",
    "cogito-v1-preview-llama-8B",
    "gemma-2-9b-it",
    "glm-4-9b-chat",
    "granite-3.3-8b-instruct",
    "internlm3-8b-instruct",
)


def load_experiment_data(unified_dir: Path, dataset: str = DATASET) -> tuple[list[str], list[str], np.ndarray]:
    """Return sorted query texts, fixed model IDs, and a complete per-instance score matrix."""
    queries = pd.read_parquet(
        unified_dir / "queries_llmrouterbench.parquet", columns=["query_id", "text"]
    )
    responses = pd.read_parquet(
        unified_dir / "responses_llmrouterbench.parquet",
        columns=["query_id", "config_id", "correct", "label_source"],
    )
    prefix = f"{dataset}/test/"
    responses = responses[
        responses["query_id"].str.startswith(prefix)
        & responses["label_source"].eq("per_record")
    ].copy()
    observed_models = set(responses["config_id"].unique())
    missing_models = set(OPEN_WEIGHT_MODELS) - observed_models
    if missing_models:
        raise ValueError(f"{dataset}: missing expected open-weight candidates: {sorted(missing_models)}")
    responses = responses[responses["config_id"].isin(OPEN_WEIGHT_MODELS)]

    query_ids = sorted(responses["query_id"].unique())
    query_lookup = queries.set_index("query_id")["text"]
    missing_queries = set(query_ids) - set(query_lookup.index)
    if missing_queries:
        raise ValueError(f"Responses reference {len(missing_queries)} absent query texts")
    model_ids = sorted(OPEN_WEIGHT_MODELS)
    matrix = responses.pivot(index="query_id", columns="config_id", values="correct")
    matrix = matrix.reindex(index=query_ids, columns=model_ids)
    if matrix.isna().any().any():
        missing = int(matrix.isna().sum().sum())
        raise ValueError(f"{dataset}: expected complete query×model coverage, found {missing} missing scores")
    scores = matrix.to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Per-instance scores must be finite values in [0, 1]")
    return [str(query_lookup.loc[qid]) for qid in query_ids], model_ids, scores


def _argmax_stable(values: np.ndarray, model_ids: list[str]) -> int:
    maximum = np.max(values)
    return min((i for i, value in enumerate(values) if value == maximum), key=lambda i: model_ids[i])


def _fit_predict(
    strategy: str,
    train_texts: list[str],
    test_texts: list[str],
    train_scores: np.ndarray,
    model_ids: list[str],
    seed: int,
) -> np.ndarray:
    """Fit a routing baseline on train outcomes only and return one model index per test query."""
    n_train, n_models = train_scores.shape
    best_global = _argmax_stable(train_scores.mean(axis=0), model_ids)

    if strategy == "random":
        return np.random.default_rng(seed).integers(0, n_models, size=len(test_texts))
    if strategy == "best_single":
        return np.full(len(test_texts), best_global, dtype=int)
    if strategy == "length_bucket":
        train_lengths = np.asarray([len(text.split()) for text in train_texts])
        test_lengths = np.asarray([len(text.split()) for text in test_texts])
        edges = np.quantile(train_lengths, [1 / 3, 2 / 3])
        train_buckets = np.digitize(train_lengths, edges)
        test_buckets = np.digitize(test_lengths, edges)
        choices = {}
        for bucket in np.unique(train_buckets):
            rows = train_buckets == bucket
            choices[int(bucket)] = _argmax_stable(train_scores[rows].mean(axis=0), model_ids)
        return np.asarray([choices.get(int(bucket), best_global) for bucket in test_buckets], dtype=int)
    if strategy in {"tfidf_knn", "tfidf_classifier"}:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30_000, sublinear_tf=True)
        train_vectors = vectorizer.fit_transform(train_texts)
        test_vectors = vectorizer.transform(test_texts)
        if strategy == "tfidf_knn":
            similarities = cosine_similarity(test_vectors, train_vectors, dense_output=True)
            k = min(15, n_train)
            neighbor_indices = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
            predictions = np.empty(len(test_texts), dtype=int)
            for i, neighbors in enumerate(neighbor_indices):
                local_scores = train_scores[neighbors].mean(axis=0)
                predictions[i] = _argmax_stable(local_scores, model_ids)
            return predictions

        train_choices = np.asarray([_argmax_stable(row, model_ids) for row in train_scores])
        if np.unique(train_choices).size == 1:
            return np.full(len(test_texts), int(train_choices[0]), dtype=int)
        classifier = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
        classifier.fit(train_vectors, train_choices)
        return classifier.predict(test_vectors).astype(int)

    raise ValueError(f"Unknown strategy: {strategy}")


def _score_choices(test_scores: np.ndarray, choices: np.ndarray) -> dict[str, float]:
    selected = test_scores[np.arange(len(choices)), choices]
    oracle = test_scores.max(axis=1)
    solvable = oracle >= 1.0
    return {
        "quality_mean": float(selected.mean()),
        "exact_success_rate": float((selected >= 1.0).mean()),
        "oracle_quality_mean": float(oracle.mean()),
        "oracle_quality_gap": float((oracle - selected).mean()),
        "solvable_query_fraction": float(solvable.mean()),
        "recall_failure_rate_on_solvable": (
            float((selected[solvable] < 1.0).mean()) if solvable.any() else float("nan")
        ),
    }


def run_experiment(unified_dir: Path, dataset: str = DATASET) -> dict[str, Any]:
    texts, model_ids, scores = load_experiment_data(unified_dir, dataset)
    strategies = ("random", "best_single", "length_bucket", "tfidf_knn", "tfidf_classifier")
    seeds = []
    for seed in SEEDS_LLMROUTERBENCH:
        train_idx, test_idx = iid_split(len(texts), test_frac=0.3, seed=seed)
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        seed_result: dict[str, Any] = {
            "seed": seed,
            "train_queries": len(train_idx),
            "test_queries": len(test_idx),
            "train_fraction": len(train_idx) / len(texts),
            "routers": {},
        }
        for strategy in strategies:
            choices = _fit_predict(
                strategy, train_texts, test_texts, scores[train_idx], model_ids, seed
            )
            seed_result["routers"][strategy] = _score_choices(scores[test_idx], choices)
        seed_result["routers"]["oracle_upper_bound"] = _score_choices(
            scores[test_idx], scores[test_idx].argmax(axis=1)
        )
        seeds.append(seed_result)

    summary = {}
    for strategy in (*strategies, "oracle_upper_bound"):
        metric_names = seeds[0]["routers"][strategy].keys()
        summary[strategy] = {
            metric: {
                "mean": float(np.nanmean([item["routers"][strategy][metric] for item in seeds])),
                "std": float(np.nanstd([item["routers"][strategy][metric] for item in seeds], ddof=1)),
            }
            for metric in metric_names
        }
    return {
        "protocol": "five seeded iid 70/30 query split; fit routers on train only; ArenaHard per-instance labels",
        "dataset": dataset,
        "queries": len(texts),
        "candidates": model_ids,
        "candidate_count": len(model_ids),
        "seeds": list(SEEDS_LLMROUTERBENCH),
        "label_semantics": "per-record varying ArenaHard labels; no aggregate_fallback rows",
        "metrics_note": "Cost and latency omitted: local-model cost_usd is zero/indistinguishable and historical latency is absent.",
        "results_by_seed": seeds,
        "summary_mean_std": summary,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unified-dir", type=Path, default=ROOT / "data" / "processed" / "unified")
    parser.add_argument("--dataset", default=DATASET, choices=[DATASET])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "phase1" / "llmrouterbench_arenahard.json")
    args = parser.parse_args(argv)

    result = run_experiment(args.unified_dir, args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Dataset: {result['dataset']} · queries={result['queries']} · candidates={result['candidate_count']}")
    for strategy, metrics in result["summary_mean_std"].items():
        print(
            f"{strategy:22s} quality={metrics['quality_mean']['mean']:.4f} "
            f"±{metrics['quality_mean']['std']:.4f} · exact={metrics['exact_success_rate']['mean']:.4f} "
            f"±{metrics['exact_success_rate']['std']:.4f} · recall-fail="
            f"{metrics['recall_failure_rate_on_solvable']['mean']:.4f}"
        )
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
