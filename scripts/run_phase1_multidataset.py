"""Audit and run the LLMRouterBench performance-only 15-dataset protocol.

Protocol follows this repository's PLAN: the official 20-model lightweight pool,
70/30 train/test split, five declared seeds, and per-dataset splits grouped by
identical prompt so all model outcomes stay on the same side. The upstream
LLMRouterBench loader documents this dataset-then-prompt grouping, while its
current generic defaults are 80/20 and seed 42; this script applies the local
PLAN's explicit 70/30 and five seeds rather than claiming a byte-identical
upstream split implementation.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from routedata.splits import SEEDS_LLMROUTERBENCH

UNIFIED = ROOT / "data" / "processed" / "unified"
OUTPUT = ROOT / "outputs" / "phase1"

# Performance-only pool from the official LLMRouterBench README, Model Pools.
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

# Official performance-only dataset IDs, sample counts, and metrics as listed in README.
DATASET_SPEC = {
    "aime": {"expected_queries": 60, "metric": "accuracy"},
    "math500": {"expected_queries": 500, "metric": "accuracy"},
    "mathbench": {"expected_queries": 150, "metric": "accuracy"},
    "humaneval": {"expected_queries": 164, "metric": "pass@1"},
    "mbpp": {"expected_queries": 974, "metric": "pass@1"},
    "livecodebench": {"expected_queries": 1055, "metric": "pass@1"},
    "bbh": {"expected_queries": 1080, "metric": "accuracy, 3-shot"},
    "korbench": {"expected_queries": 1250, "metric": "accuracy, 3-shot"},
    "kandk": {"expected_queries": 700, "metric": "accuracy"},
    "mmlupro": {"expected_queries": 1000, "metric": "accuracy"},
    "gpqa": {"expected_queries": 198, "metric": "accuracy"},
    "finqa": {"expected_queries": 1147, "metric": "accuracy"},
    "medqa": {"expected_queries": 1273, "metric": "accuracy"},
    "emorynlp": {"expected_queries": 697, "metric": "accuracy"},
    "meld": {"expected_queries": 1232, "metric": "accuracy"},
}

ROUTER_NAMES = (
    "random",
    "best_single",
    "length_bucket",
    "tfidf_knn",
    "tfidf_classifier",
    "tfidf_score_regressor",
    "dataset_oracle_upper_bound",
)
FITTED_ROUTER_NAMES = tuple(name for name in ROUTER_NAMES if name != "dataset_oracle_upper_bound")

UPSTREAM_SOURCES = {
    "README": "https://github.com/ynulihao/LLMRouterBench/blob/main/README.md#L66-L82",
    "model_pool": "https://github.com/ynulihao/LLMRouterBench/blob/main/README.md#L306-L329",
    "dataset_table": "https://github.com/ynulihao/LLMRouterBench/blob/main/README.md#L261-L291",
    "split_documentation": "https://github.com/ynulihao/LLMRouterBench/blob/main/baselines/README.md#L113-L135",
    "split_implementation": "https://github.com/ynulihao/LLMRouterBench/blob/main/baselines/data_loader.py#L342-L449",
    "official_aggregator_defaults": "https://github.com/ynulihao/LLMRouterBench/blob/main/baselines/aggregators.py#L714-L728",
}


def _index_from_query_id(query_id: str) -> tuple[int, str]:
    tail = str(query_id).rsplit("/", 1)[-1]
    match = re.match(r"(\d+)", tail)
    return (int(match.group(1)), str(query_id)) if match else (2**63 - 1, str(query_id))


def split_dataset_by_prompt(
    query_ids: list[str],
    texts: list[str],
    train_ratio: float,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Split one dataset by unique prompt, mirroring upstream leakage prevention.

    Prompt groups are ordered by the minimum numeric source index; a Python
    Random instance shuffles prompt positions, then the first floor(N*ratio)
    groups are assigned to train. All rows sharing prompt text remain together.
    """
    if len(query_ids) != len(texts):
        raise ValueError("query_ids and texts must have the same length")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between zero and one")

    prompt_rows: dict[str, list[int]] = defaultdict(list)
    prompt_min_index: dict[str, tuple[int, str]] = {}
    for row_index, (query_id, text) in enumerate(zip(query_ids, texts, strict=False)):
        prompt = str(text).strip()
        prompt_rows[prompt].append(row_index)
        idx = _index_from_query_id(query_id)
        prompt_min_index[prompt] = min(prompt_min_index.get(prompt, idx), idx)

    prompts = sorted(prompt_rows, key=lambda prompt: prompt_min_index[prompt])
    shuffled_positions = list(range(len(prompts)))
    if rng is None:
        if seed is None:
            raise ValueError("Pass either seed or rng")
        rng = random.Random(seed)
    rng.shuffle(shuffled_positions)
    n_train_prompts = int(len(prompts) * train_ratio)
    train_prompt_set = {prompts[index] for index in shuffled_positions[:n_train_prompts]}

    train_rows = [i for prompt in prompts if prompt in train_prompt_set for i in prompt_rows[prompt]]
    test_rows = [i for prompt in prompts if prompt not in train_prompt_set for i in prompt_rows[prompt]]
    train_idx = np.asarray(sorted(train_rows), dtype=int)
    test_idx = np.asarray(sorted(test_rows), dtype=int)
    if not len(train_idx) or not len(test_idx):
        raise ValueError("Split produced an empty train or test set")
    return (
        train_idx,
        test_idx,
        {
            "unique_prompts": len(prompts),
            "train_prompts": n_train_prompts,
            "test_prompts": len(prompts) - n_train_prompts,
        },
    )


def audit_data(
    unified_dir: Path = UNIFIED,
) -> tuple[dict[str, Any], dict[str, tuple[list[str], list[str], np.ndarray]]]:
    """Audit local pool, expected tasks, labels, score variation, and matrix coverage."""
    queries = pd.read_parquet(
        unified_dir / "queries_llmrouterbench.parquet",
        columns=["query_id", "text", "prompt", "prompt_variant_count"],
    )
    responses = pd.read_parquet(
        unified_dir / "responses_llmrouterbench.parquet",
        columns=["query_id", "config_id", "correct", "label_source"],
    )
    queries = queries.copy()
    queries["dataset"] = queries["query_id"].str.split("/").str[0]
    responses = responses.copy()
    responses["dataset"] = responses["query_id"].str.split("/").str[0]

    found_models = set(responses["config_id"].unique())
    missing_models = sorted(set(OPEN_WEIGHT_MODELS) - found_models)
    _extra_models_in_selected_data: list[str] = []
    experiment_data: dict[str, tuple[list[str], list[str], np.ndarray]] = {}
    dataset_audit: dict[str, Any] = {}
    global_excluded_fallbacks: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []

    if missing_models:
        errors.append(f"Missing official pool models: {missing_models}")

    query_text = queries.set_index("query_id")["prompt"]
    for dataset, spec in DATASET_SPEC.items():
        subset = responses[
            responses["dataset"].eq(dataset) & responses["config_id"].isin(OPEN_WEIGHT_MODELS)
        ].copy()
        fallback_qids = sorted(
            subset.loc[subset["label_source"].eq("aggregate_fallback"), "query_id"].unique()
        )
        global_excluded_fallbacks.extend(fallback_qids)
        # Exclude an entire query if any candidate score was aggregate-derived;
        # this preserves a common, complete 20-model candidate matrix.
        clean = subset[~subset["query_id"].isin(fallback_qids)].copy()
        pivot = clean.pivot(index="query_id", columns="config_id", values="correct")
        pivot = pivot.reindex(columns=sorted(OPEN_WEIGHT_MODELS))
        complete_mask = pivot.notna().all(axis=1)
        incomplete_qids = sorted(pivot.index[~complete_mask].astype(str).tolist())
        pivot = pivot.loc[complete_mask]
        query_ids = sorted(pivot.index.astype(str).tolist(), key=_index_from_query_id)
        text_values = [str(query_text.loc[qid]) for qid in query_ids]
        score_matrix = pivot.reindex(index=query_ids, columns=sorted(OPEN_WEIGHT_MODELS)).to_numpy(
            dtype=float
        )
        if not np.isfinite(score_matrix).all() or ((score_matrix < 0) | (score_matrix > 1)).any():
            errors.append(f"{dataset}: scores must be finite and in [0,1]")
        if pivot.empty:
            errors.append(f"{dataset}: no complete per-instance query rows")
        if len(set(clean["config_id"].unique())) != len(OPEN_WEIGHT_MODELS):
            errors.append(f"{dataset}: does not contain all 20 official candidate models")

        per_model_variation = clean.groupby("config_id")["correct"].nunique()
        constant_score_models = sorted(per_model_variation[per_model_variation == 1].index.tolist())
        observed_before_exclusion = int(subset["query_id"].nunique())
        actual_n = len(query_ids)
        if actual_n != spec["expected_queries"]:
            warnings.append(
                f"{dataset}: official README says {spec['expected_queries']} prompts; "
                f"local complete per-instance matrix has {actual_n} after excluding {len(fallback_qids)} aggregate-labeled prompt(s)"
            )
        duplicate_prompt_groups = int(pd.Series(text_values).duplicated(keep=False).sum())
        dataset_audit[dataset] = {
            "official_expected_prompts": spec["expected_queries"],
            "official_metric": spec["metric"],
            "local_query_rows_before_label_filter": observed_before_exclusion,
            "complete_per_instance_queries": actual_n,
            "candidate_models": int(clean["config_id"].nunique()),
            "matrix_rows": int(score_matrix.shape[0]),
            "matrix_columns": int(score_matrix.shape[1]),
            "aggregate_fallback_query_ids_excluded": fallback_qids,
            "other_incomplete_query_ids_excluded": incomplete_qids,
            "duplicate_prompt_rows_grouped_for_split": duplicate_prompt_groups,
            "queries_with_multiple_source_prompts": int(
                queries.loc[queries["query_id"].isin(query_ids), "prompt_variant_count"].gt(1).sum()
            ),
            "models_with_constant_observed_scores": constant_score_models,
        }
        experiment_data[dataset] = (query_ids, text_values, score_matrix)

    return {
        "protocol": {
            "pool": "LLMRouterBench official Performance Setting (20 lightweight models)",
            "datasets": "15 official performance-only datasets",
            "split": "70/30 independently per dataset, grouped by exact prompt to prevent leakage",
            "seeds": [42, 999, 2024, 2025, 3407],
            "seed_provenance": "Five seeds required by the repository PLAN/catalog; upstream current generic API documents seed as configurable and defaults to 42, so this is not claimed to be the upstream paper's exact seed set.",
            "label_policy": "exclude entire query if any candidate label is aggregate_fallback; preserve only complete 20-model per-instance rows",
            "upstream_note": "Upstream documents prompt-level grouping; current generic API defaults are configurable and often 80/20, seed 42. The local PLAN's 70/30 and five seeds are applied here.",
        },
        "sources": UPSTREAM_SOURCES,
        "candidate_count": len(OPEN_WEIGHT_MODELS),
        "candidate_models": list(OPEN_WEIGHT_MODELS),
        "missing_candidate_models": missing_models,
        "datasets": dataset_audit,
        "total_complete_queries": sum(len(v[0]) for v in experiment_data.values()),
        "excluded_aggregate_fallback_query_ids": sorted(set(global_excluded_fallbacks)),
        "warnings": warnings,
        "errors": errors,
    }, experiment_data


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
        choices: dict[int, int] = {}
        for bucket in np.unique(train_buckets):
            choices[int(bucket)] = _argmax_stable(
                train_scores[train_buckets == bucket].mean(axis=0), model_ids
            )
        return np.asarray([choices.get(int(bucket), best_global) for bucket in test_buckets], dtype=int)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30_000, sublinear_tf=True)
    train_vectors = vectorizer.fit_transform(train_texts)
    test_vectors = vectorizer.transform(test_texts)
    if strategy == "tfidf_knn":
        similarities = cosine_similarity(test_vectors, train_vectors, dense_output=True)
        k = min(15, n_train)
        neighbor_indices = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
        return np.asarray(
            [
                _argmax_stable(train_scores[neighbors].mean(axis=0), model_ids)
                for neighbors in neighbor_indices
            ],
            dtype=int,
        )
    if strategy == "tfidf_classifier":
        train_choices = np.asarray([_argmax_stable(row, model_ids) for row in train_scores])
        if np.unique(train_choices).size == 1:
            return np.full(len(test_texts), int(train_choices[0]), dtype=int)
        classifier = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
        classifier.fit(train_vectors, train_choices)
        return classifier.predict(test_vectors).astype(int)
    if strategy == "tfidf_score_regressor":
        regressor = Ridge(alpha=1.0)
        regressor.fit(train_vectors, train_scores)
        predictions = np.asarray(regressor.predict(test_vectors))
        return np.argmax(predictions, axis=1).astype(int)
    raise ValueError(f"Unknown strategy: {strategy}")


def _score_choices(test_scores: np.ndarray, choices: np.ndarray) -> dict[str, float]:
    selected = test_scores[np.arange(len(choices)), choices]
    oracle = test_scores.max(axis=1)
    solvable = oracle >= 1.0
    return {
        "avg_score": float(selected.mean()),
        "exact_success_rate": float((selected >= 1.0).mean()),
        "oracle_avg_score": float(oracle.mean()),
        "absolute_oracle_score_gap": float((oracle - selected).mean()),
        "solvable_query_fraction": float(solvable.mean()),
        "recall_failure_rate_on_solvable": float((selected[solvable] < 1.0).mean())
        if solvable.any()
        else float("nan"),
        "solvable_query_count": float(solvable.sum()),
        "recall_failure_count": float((selected[solvable] < 1.0).sum()),
    }


def _summarize_seed(
    dataset_results: dict[str, dict[str, dict[str, float]]],
    dataset_sizes: dict[str, int],
    strategies: tuple[str, ...],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    _total_rows = sum(dataset_sizes.values())
    for strategy in strategies:
        metric_names = next(iter(next(iter(dataset_results.values())).values())).keys()
        output[strategy] = {}
        for metric in metric_names:
            per_dataset = [dataset_results[ds][strategy][metric] for ds in dataset_results]
            weights = np.asarray([dataset_sizes[ds] for ds in dataset_results], dtype=float)
            weighted = float(np.average(per_dataset, weights=weights))
            output[strategy][metric] = {
                "macro_dataset_average": float(np.nanmean(per_dataset)),
                "sample_weighted_average": weighted,
            }
        # Relative gains and oracle gap in the convention stated in upstream README.
        macro_scores = {ds: dataset_results[ds][strategy]["avg_score"] for ds in dataset_results}
        random_scores = {ds: dataset_results[ds]["random"]["avg_score"] for ds in dataset_results}
        best_scores = {ds: dataset_results[ds]["best_single"]["avg_score"] for ds in dataset_results}
        oracle_scores = {ds: dataset_results[ds]["oracle_upper_bound"]["avg_score"] for ds in dataset_results}
        gain_random = [
            macro_scores[ds] / random_scores[ds] - 1 for ds in dataset_results if random_scores[ds] > 0
        ]
        gain_best = [macro_scores[ds] / best_scores[ds] - 1 for ds in dataset_results if best_scores[ds] > 0]
        gap_oracle = [
            1 - macro_scores[ds] / oracle_scores[ds] for ds in dataset_results if oracle_scores[ds] > 0
        ]
        output[strategy]["gain_vs_random_macro"] = (
            float(np.mean(gain_random)) if gain_random else float("nan")
        )
        output[strategy]["gain_vs_best_single_macro"] = (
            float(np.mean(gain_best)) if gain_best else float("nan")
        )
        output[strategy]["gap_to_oracle_macro"] = float(np.mean(gap_oracle)) if gap_oracle else float("nan")
        solvable_n = sum(dataset_results[ds][strategy]["solvable_query_count"] for ds in dataset_results)
        failures_n = sum(dataset_results[ds][strategy]["recall_failure_count"] for ds in dataset_results)
        output[strategy]["recall_failure_rate_pooled_solvable"] = (
            failures_n / solvable_n if solvable_n else float("nan")
        )
    return output


def run_evaluation(
    experiment_data: dict[str, tuple[list[str], list[str], np.ndarray]],
    train_ratio: float = 0.7,
    seeds: tuple[int, ...] = SEEDS_LLMROUTERBENCH,
) -> dict[str, Any]:
    strategies = (*ROUTER_NAMES, "oracle_upper_bound")
    dataset_sizes = {ds: len(values[0]) for ds, values in experiment_data.items()}
    seed_outputs: list[dict[str, Any]] = []
    for seed in seeds:
        split_rng = random.Random(seed)
        per_dataset: dict[str, dict[str, dict[str, float]]] = {}
        split_audit: dict[str, Any] = {}
        for dataset in sorted(experiment_data):
            query_ids, texts, scores = experiment_data[dataset]
            train_idx, test_idx, prompt_counts = split_dataset_by_prompt(
                query_ids, texts, train_ratio, rng=split_rng
            )
            split_audit[dataset] = {
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "train_prompt_groups": prompt_counts["train_prompts"],
                "test_prompt_groups": prompt_counts["test_prompts"],
            }
            train_texts = [texts[i] for i in train_idx]
            test_texts = [texts[i] for i in test_idx]
            per_dataset[dataset] = {}
            for strategy in FITTED_ROUTER_NAMES:
                # Random router gets distinct but deterministic random streams by dataset.
                strategy_seed = seed + sorted(experiment_data).index(dataset) * 100_003
                choice = _fit_predict(
                    strategy,
                    train_texts,
                    test_texts,
                    scores[train_idx],
                    list(sorted(OPEN_WEIGHT_MODELS)),
                    strategy_seed,
                )
                per_dataset[dataset][strategy] = _score_choices(scores[test_idx], choice)
            dataset_best_model = _argmax_stable(
                scores[test_idx].mean(axis=0), list(sorted(OPEN_WEIGHT_MODELS))
            )
            per_dataset[dataset]["dataset_oracle_upper_bound"] = _score_choices(
                scores[test_idx], np.full(len(test_idx), dataset_best_model, dtype=int)
            )
            per_dataset[dataset]["oracle_upper_bound"] = _score_choices(
                scores[test_idx], scores[test_idx].argmax(axis=1)
            )

        seed_outputs.append(
            {
                "seed": seed,
                "dataset_splits": split_audit,
                "summary": _summarize_seed(per_dataset, dataset_sizes, strategies),
                "by_dataset": per_dataset,
            }
        )

    aggregate: dict[str, Any] = {}
    for strategy in strategies:
        aggregate[strategy] = {}
        metrics = seed_outputs[0]["summary"][strategy]
        for metric, first_value in metrics.items():
            if isinstance(first_value, dict):
                aggregate[strategy][metric] = {}
                for sub_metric in first_value:
                    values = np.asarray(
                        [item["summary"][strategy][metric][sub_metric] for item in seed_outputs],
                        dtype=float,
                    )
                    aggregate[strategy][metric][sub_metric] = {
                        "mean": float(np.nanmean(values)),
                        "std_between_seeds": float(np.nanstd(values, ddof=1)),
                        "per_seed": values.tolist(),
                    }
            else:
                values = np.asarray([item["summary"][strategy][metric] for item in seed_outputs], dtype=float)
                aggregate[strategy][metric] = {
                    "mean": float(np.nanmean(values)),
                    "std_between_seeds": float(np.nanstd(values, ddof=1)),
                    "per_seed": values.tolist(),
                }

    return {
        "protocol": {
            "pool": "LLMRouterBench official Performance Setting (20 models)",
            "datasets": list(sorted(experiment_data)),
            "split_ratio_train": train_ratio,
            "split_unit": "exact prompt within each dataset; duplicate prompt rows remain in the same split",
            "split_algorithm": "sort prompts by minimum source index; seeded Python random shuffle within each dataset; first floor(n_prompts*0.7) prompts train",
            "seeds": list(seeds),
            "routers": list(FITTED_ROUTER_NAMES),
            "reference_baselines": [
                "random",
                "best_single",
                "dataset_oracle_upper_bound",
                "oracle_upper_bound",
            ],
            "oracle_note": "Best Single is train-selected. Dataset Oracle uses held-out test labels to select one best model per dataset and is a reference upper bound, not a deployable router. Instance Oracle is the per-query upper bound.",
            "labels": "per-record score; exclude whole MBPP query mbpp/test/665 because one model has aggregate_fallback; retain complete 20-model matrix otherwise",
            "upstream_comparability": "Matches the official pool, tasks, per-dataset prompt grouping, and leakage prevention. Uses local PLAN's five seeds and 70/30 rather than upstream generic defaults (often one seed / 80-20); the upstream repository uses configurable seeds and split ratios. Local router algorithms are baseline comparators, not claimed reproductions of five published learned routers.",
            "metric_definitions": {
                "avg_score": "mean normalized source score (accuracy/pass@1/judge score) of selected model",
                "macro_dataset_average": "unweighted mean of the 15 per-dataset metrics",
                "sample_weighted_average": "per-dataset metric weighted by query count",
                "gain_vs_random_macro": "mean over datasets of router_avg_score / random_avg_score - 1",
                "gain_vs_best_single_macro": "mean over datasets of router_avg_score / train-selected best-single_avg_score - 1",
                "gap_to_oracle_macro": "mean over datasets of 1 - router_avg_score / per-query oracle_avg_score",
                "recall_failure_rate_on_solvable": "fraction of queries where at least one candidate scores 1 but selected candidate scores below 1",
            },
        },
        "query_count": sum(dataset_sizes.values()),
        "candidate_count": len(OPEN_WEIGHT_MODELS),
        "summary_mean_std_across_seeds": aggregate,
        "results_by_seed": seed_outputs,
    }


def _json_safe(value: Any) -> Any:
    """Convert NumPy scalars and non-finite floats to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unified-dir", type=Path, default=UNIFIED)
    parser.add_argument("--audit-output", type=Path, default=OUTPUT / "protocol_audit.json")
    parser.add_argument("--output", type=Path, default=OUTPUT / "llmrouterbench_15datasets.json")
    parser.add_argument(
        "--audit-only", action="store_true", help="Write protocol/data audit without fitting routers"
    )
    args = parser.parse_args(argv)

    audit, experiment_data = audit_data(args.unified_dir)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(_json_safe(audit), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(
        f"Audit: {audit['candidate_count']} candidates · {len(audit['datasets'])} datasets · "
        f"{audit['total_complete_queries']} complete per-instance queries"
    )
    for dataset, item in audit["datasets"].items():
        print(
            f"  {dataset:16s} {item['complete_per_instance_queries']:5d}/"
            f"{item['official_expected_prompts']:<4d} queries · {item['candidate_models']} models · "
            f"fallback excluded={len(item['aggregate_fallback_query_ids_excluded'])}"
        )
    for warning in audit["warnings"]:
        print(f"WARNING: {warning}")
    if audit["errors"]:
        for error in audit["errors"]:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    if args.audit_only:
        print(f"Audit written to {args.audit_output}")
        return

    result = run_evaluation(experiment_data)
    result["audit_file"] = (
        str(args.audit_output.relative_to(ROOT))
        if args.audit_output.is_relative_to(ROOT)
        else str(args.audit_output)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_safe(result), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print("\nMacro dataset average (mean ± SD across seeds):")
    for strategy, values in result["summary_mean_std_across_seeds"].items():
        avg = values["avg_score"]["macro_dataset_average"]["mean"]
        std = values["avg_score"]["macro_dataset_average"]["std_between_seeds"]
        gap = values["gap_to_oracle_macro"]["mean"]
        recall = values["recall_failure_rate_pooled_solvable"]["mean"]
        print(f"{strategy:24s} AvgScore={avg:.4f} ±{std:.4f} · Gap@O={gap:.4f} · recall-failure={recall:.4f}")
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
