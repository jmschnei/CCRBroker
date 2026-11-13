"""SPROUT adapter for wide-format parquet data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class SproutAdapter:
    """Adapter for SPROUT benchmark (parquet, wide format).

    SPROUT structure:
    - prompt: question text
    - golden_answer: gold answer
    - <model_name>: dict with {score, response, num_input_tokens, num_output_tokens, judge_response}
    """

    name = "sprout"

    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir

    def detect(self) -> bool:
        """Check if SPROUT directory exists."""
        path = self.raw_dir / "data" / "raw" / "sprout"
        return path.exists()

    @staticmethod
    def _detect_model_columns(df: pd.DataFrame) -> list[str]:
        """Detect which columns are model response columns.

        SPROUT model columns contain dicts with 'score' or 'response' keys.
        """
        known_meta = {
            "key",
            "dataset",
            "dataset_level",
            "dataset_idx",
            "prompt",
            "golden_answer",
            "answer",
            "gold_answer",
            "query",
            "question",
        }
        model_cols = []
        for col in df.columns:
            if col.lower() in known_meta:
                continue
            sample = df[col].dropna().head(5)
            if len(sample) == 0:
                continue
            # Check if it's a dict with 'score' or 'response' key
            if sample.apply(lambda x: isinstance(x, dict) and ("score" in x or "response" in x)).any():
                model_cols.append(col)
        return model_cols

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load SPROUT data into the unified schema."""
        raw_dir = self.raw_dir / "data" / "raw" / "sprout"

        if not raw_dir.exists():
            raise FileNotFoundError(f"SPROUT data not found at: {raw_dir}")

        # Find all parquet files
        parquet_files = sorted(raw_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in: {raw_dir}")

        # Load all parquet files
        dfs = []
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)

        # Detect model columns
        model_cols = self._detect_model_columns(df)

        if not model_cols:
            raise ValueError(f"No model columns detected in SPROUT. Columns: {list(df.columns)}")

        # Build queries, configs, and responses tables
        query_ids = []
        prompts = []
        golds = []
        configs = []
        responses = []

        for idx, row in df.iterrows():
            qid = str(row.get("key", f"{idx}"))
            prompt = str(row.get("prompt", ""))
            gold = str(row.get("golden_answer", row.get("answer", row.get("gold_answer", ""))))

            query_ids.append(qid)
            prompts.append(prompt)
            golds.append(gold)

            # Create a config and response for each model column
            for model in model_cols:
                cell_value = row.get(model)

                # Skip if cell is empty or NaN
                if cell_value is None or pd.isna(cell_value):
                    continue

                # Extract from dict structure
                if isinstance(cell_value, dict):
                    response_text = cell_value.get("response", "")
                    score = cell_value.get("score", 0.0)
                else:
                    # Fallback: treat as string
                    response_text = str(cell_value)
                    score = 1.0

                # Create a config for this model
                config = {"id": f"sprout-{model}", "name": model, "kind": "candidate"}
                configs.append(config)

                # Create a response row
                responses.append(
                    {
                        "id": f"{qid}:{model}",
                        "query_id": qid,
                        "config_id": f"sprout-{model}",
                        "prompt": prompt,
                        "gold": gold,
                        "correct": score >= 0.5,  # Binary correctness
                        "score": float(score),
                        "response": response_text,
                        "latency_ms": None,
                    }
                )

        queries = pd.DataFrame({"id": query_ids, "prompt": prompts, "gold": golds})

        # Normalizar esquema para que coincida con los otros datasets
        queries = queries.rename(columns={"id": "query_id", "prompt": "text", "gold": "answer"})

        # Deduplicate configs
        config_df = pd.DataFrame(configs)
        if not config_df.empty:
            config_df = config_df.drop_duplicates(subset=["id"])

        # Normalizar esquema de configs
        config_df = config_df.rename(columns={"id": "config_id", "name": "model", "kind": "kind"})

        return queries, config_df, pd.DataFrame(responses)


def create_sprout_adapter(raw_dir: Path) -> SproutAdapter:
    """Create a SPROUT adapter."""
    return SproutAdapter(raw_dir)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    adapter = create_sprout_adapter(raw_dir)
    queries, configs, responses = adapter.load()
    print(f"SPROUT: {len(queries)} queries, {len(configs)} configs, {len(responses)} responses")
    print(f"Sample config: {configs.iloc[0].to_dict() if len(configs) > 0 else 'None'}")
    print(f"Sample response: {responses.iloc[0].to_dict() if len(responses) > 0 else 'None'}")
