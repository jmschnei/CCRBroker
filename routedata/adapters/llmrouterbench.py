"""LLMRouterBench → unified schema adapter."""

import json
import sys
from pathlib import Path

import pandas as pd

from .base import wide_to_tables


class LLMRouterBenchAdapter:
    """LLMRouterBench adapter.

    Input: data/raw/llmrouterbench/extracted/bench-release/<dataset>/<split>/<model>/*.json
    Each JSON contains:
      - performance: float (aggregate)
      - model_name: str
      - dataset_name: str
      - records: list of dicts with index, instance_id, origin_query, etc.

    Strategy:
    1. Group records by (dataset, split, query_id)
    2. Build wide table: one row per query, columns per model with scores
    3. Use wide_to_tables to convert to unified schema
    """

    def load(self, path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(f"LLMRouterBench root not found: {root}")

        # Find all JSON files
        json_files = list(root.rglob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {root}")

        print(f"Processing {len(json_files)} JSON files...")

        # First pass: collect all records grouped by query
        # query_key = (dataset, split, instance_id)
        # For each query, collect scores from different models
        queries_data: dict[tuple, dict] = {}
        model_scores: dict[tuple, dict[str, float]] = {}

        for i, json_file in enumerate(json_files):
            if (i + 1) % 100 == 0:
                print(f"  Read {i + 1}/{len(json_files)} files...")

            try:
                with open(json_file) as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  Warning: could not read {json_file}: {e}")
                continue

            model_name = data.get("model_name", "unknown")
            dataset_name = data.get("dataset_name", "unknown")
            split = data.get("split", "unknown")
            records = data.get("records", [])

            for rec in records:
                instance_id = rec.get("instance_id") or rec.get("index", 0)
                query_key = (dataset_name, split, instance_id)

                # Get prompt
                prompt = rec.get("origin_query") or rec.get("query") or ""

                if query_key not in queries_data:
                    queries_data[query_key] = {
                        "query_id": f"{dataset_name}/{split}/{instance_id}",
                        "text": prompt,
                        "domain": f"{dataset_name}/{split}",
                        "answer": "",
                    }
                    model_scores[query_key] = {}

                # Get score for this model
                score = rec.get("correct") or rec.get("performance") or data.get("performance")
                if score is not None:
                    model_scores[query_key][model_name] = float(score)

        if not queries_data:
            raise ValueError("No records found in LLMRouterBench data")

        print(f"  Found {len(queries_data)} unique queries")
        print(f"  Found {len(set(m for scores in model_scores.values() for m in scores.keys()))} models")

        # Build wide DataFrame
        rows = []
        for (dataset_name, split, instance_id), qdata in queries_data.items():
            row = {
                "query_id": qdata["query_id"],
                "prompt": qdata["text"],
                "domain": qdata["domain"],
                "answer": qdata["answer"],
            }
            # Add score columns for all models
            scores = model_scores[(dataset_name, split, instance_id)]
            for model_name in scores:
                row[model_name] = scores[model_name]
            rows.append(row)

        df = pd.DataFrame(rows)

        # Find all model columns
        model_cols = [col for col in df.columns if col not in ["query_id", "prompt", "domain", "answer"]]

        if not model_cols:
            raise ValueError("No model columns found in LLMRouterBench data")

        print(f"  Model columns: {model_cols[:10]}{'...' if len(model_cols) > 10 else ''}")

        # Convert to unified tables
        queries, configs, responses = wide_to_tables(
            df,
            prompt_col="prompt",
            perf_cols=model_cols,
            id_col="query_id",
            domain_col="domain",
            source_name="llmrouterbench",
            answer_col="answer",
        )

        return queries, configs, responses


def _run(path: str | None) -> None:
    if path is None:
        root = Path(__file__).resolve().parents[2]
        path = str(root / "data" / "raw" / "llmrouterbench")

    adapter = LLMRouterBenchAdapter()
    queries, configs, responses = adapter.load(path)

    print(f"\nLLMRouterBench: {len(queries)} queries, {len(configs)} configs, {len(responses)} responses")
    models = sorted(responses["config_id"].dropna().unique())[:20]
    print("Models:", models)


if __name__ == "__main__":
    _run(sys.argv[1] if len(sys.argv) > 1 else None)
