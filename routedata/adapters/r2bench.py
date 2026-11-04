"""R2-Bench adapter for CSV data with judge scores.

Note: R2-Bench is very large (24GB). This adapter processes it in chunks
or can be limited to a subset of models.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class R2BenchAdapter:
    """Adapter for R2-Bench (CSV format with judge scores).
    
    R2-Bench structure (long format, organized by model):
    - data/raw/r2bench/data/<org>/<model>/<N>_judge.csv
    - prompts_id: query ID
    - key: query key
    - original_prompt: question text
    - golden_answer: gold answer
    - response: model response text
    - correctness_score: score (0.0-1.0)
    
    Model name is derived from the file path.
    """

    name = "r2bench"

    def __init__(self, raw_dir: Path, limit_models: int = 0, limit_files: int = 0):
        """Initialize adapter.
        
        Args:
            raw_dir: Root directory
            limit_models: Max number of models to process (0 = all)
            limit_files: Max number of CSV files to process (0 = all)
        """
        self.raw_dir = raw_dir
        self.limit_models = limit_models
        self.limit_files = limit_files

    def detect(self) -> bool:
        """Check if R2-Bench directory exists."""
        path = self.raw_dir / "data" / "raw" / "r2bench"
        return path.exists()

    @staticmethod
    def _extract_model_name(file_path: Path, raw_base: Path) -> str:
        """Extract model name from file path.
        
        Example: data/raw/r2bench/data/Qwen/Qwen3-235B-A22B-Instruct-2507/100_judge.csv
        Returns: "Qwen3-235B-A22B-Instruct-2507"
        """
        try:
            rel_path = file_path.relative_to(raw_base)
            parts = rel_path.parts
            # Structure: data/<org>/<model>/<file>.csv
            if len(parts) >= 3:
                return parts[2]
            return file_path.parent.name
        except ValueError:
            return file_path.parent.name

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load R2-Bench data into the unified schema."""
        raw_dir = self.raw_dir / "data" / "raw" / "r2bench"
        
        if not raw_dir.exists():
            raise FileNotFoundError(f"R2-Bench data not found at: {raw_dir}")
        
        # Find all CSV files
        csv_files = list(raw_dir.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in: {raw_dir}")
        
        # Sort by model for consistency
        csv_files = sorted(csv_files, key=lambda x: (self._extract_model_name(x, raw_dir), x.name))
        
        # Apply limits
        if self.limit_files > 0:
            csv_files = csv_files[:self.limit_files]
        
        # Group by model to apply limit_models
        models = {}
        for cf in csv_files:
            model = self._extract_model_name(cf, raw_dir)
            if model not in models:
                models[model] = []
            models[model].append(cf)
        
        if self.limit_models > 0:
            models = dict(list(models.items())[:self.limit_models])
            csv_files = [f for files in models.values() for f in files]
        
        print(f"Processing {len(csv_files)} CSV files from {len(models)} models")
        
        # Load CSV files one by one and process
        all_dfs = []
        for i, cf in enumerate(csv_files):
            try:
                df = pd.read_csv(cf)
                if not df.empty:
                    model_name = self._extract_model_name(cf, raw_dir)
                    df["_model"] = model_name
                    all_dfs.append(df)
                    if (i + 1) % 10 == 0:
                        print(f"  Read {i + 1}/{len(csv_files)} files...")
            except Exception as e:
                print(f"Warning: Could not read {cf}: {e}")
        
        if not all_dfs:
            raise ValueError(f"No valid CSV data found in: {raw_dir}")
        
        df = pd.concat(all_dfs, ignore_index=True)
        del all_dfs  # Free memory
        
        # Build queries, configs, and responses tables
        query_ids = []
        prompts = []
        golds = []
        configs = []
        responses = []
        
        seen_configs = set()
        
        for idx, row in df.iterrows():
            # Get query ID
            if "prompts_id" in row and pd.notna(row.get("prompts_id")):
                qid = str(row["prompts_id"])
            elif "key" in row and pd.notna(row.get("key")):
                qid = str(row["key"])
            else:
                qid = f"{idx}"
            
            prompt = str(row.get("original_prompt", row.get("prompt", "")))
            gold = str(row.get("golden_answer", row.get("answer", row.get("gold_answer", ""))))
            response_text = str(row.get("response", ""))
            model = str(row.get("_model", "unknown"))
            
            # Get score (may be missing)
            score = row.get("correctness_score")
            if pd.isna(score) or score is None:
                score = 0.0
            
            query_ids.append(qid)
            prompts.append(prompt)
            golds.append(gold)
            
            # Create config if not seen
            config_id = f"r2bench-{model}"
            if config_id not in seen_configs:
                seen_configs.add(config_id)
                config = {"config_id": config_id, "model": model, "temperature": 1.0, "max_tokens": 0, "source": "r2bench"}
                configs.append(config)
            
            # Create response row
            responses.append({
                "query_id": qid,
                "config_id": config_id,
                "score": float(score),
                "reward": float(score),
                "latency_ms": 0.0,
            })
        
        queries = pd.DataFrame({"query_id": query_ids, "text": prompts, "answer": golds})
        queries = queries.drop_duplicates(subset=["query_id"])
        
        config_df = pd.DataFrame(configs)
        
        return queries, config_df, pd.DataFrame(responses)


def create_r2bench_adapter(raw_dir: Path, limit_models: int = 0, limit_files: int = 0) -> R2BenchAdapter:
    """Create an R2-Bench adapter."""
    return R2BenchAdapter(raw_dir, limit_models, limit_files)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    
    # Process limited subset for testing
    adapter = create_r2bench_adapter(raw_dir, limit_models=2, limit_files=10)
    queries, configs, responses = adapter.load()
    print(f"\nR2-Bench: {len(queries)} queries, {len(configs)} configs, {len(responses)} responses")
    if len(configs) > 0:
        print(f"Models: {configs['model'].tolist()}")
    if len(responses) > 0:
        sample = responses.iloc[0].to_dict()
        print(f"Sample response: {str(sample)[:500]}...")
