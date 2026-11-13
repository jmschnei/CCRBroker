"""Pipeline: carga todos los datasets y genera el esquema unificado.

Uso:
    python -m routedata.pipeline load
    python -m routedata.pipeline load --output data/processed/unified/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from .adapters.routerbench import load as load_routerbench
from .adapters.sprout import SproutAdapter, create_sprout_adapter
from .adapters.r2bench import R2BenchAdapter
from .adapters.llmrouterbench import LLMRouterBenchAdapter


class Pipeline:
    """Carga todos los datasets y genera el esquema unificado."""

    def __init__(self, data_root: str | None = None):
        root = Path(data_root) if data_root else Path(__file__).resolve().parents[1]
        self.raw_root = root / "data" / "raw"
        self.processed_root = root / "data" / "processed" / "unified"

    def load_all(self) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Carga todos los datasets y devuelve (queries, configs, responses) por dataset."""
        results = {}

        # RouterBench
        print("\n" + "=" * 60)
        print("Loading RouterBench...")
        print("=" * 60)
        try:
            results["routerbench"] = load_routerbench(str(self.raw_root / "routerbench"))
        except Exception as e:
            print(f"  ERROR: {e}")

        # SPROUT
        print("\n" + "=" * 60)
        print("Loading SPROUT...")
        print("=" * 60)
        try:
            adapter = create_sprout_adapter(Path("."))
            results["sprout"] = adapter.load()
        except Exception as e:
            print(f"  ERROR: {e}")

        # R2-Bench
        print("\n" + "=" * 60)
        print("Loading R2-Bench...")
        print("=" * 60)
        try:
            adapter = R2BenchAdapter(raw_dir=Path("."))
            results["r2bench"] = adapter.load()
        except Exception as e:
            print(f"  ERROR: {e}")

        # LLMRouterBench
        print("\n" + "=" * 60)
        print("Loading LLMRouterBench...")
        print("=" * 60)
        try:
            adapter = LLMRouterBenchAdapter()
            results["llmrouterbench"] = adapter.load(str(self.raw_root / "llmrouterbench"))
        except Exception as e:
            print(f"  ERROR: {e}")

        return results

    def save(self, results: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
             output_dir: str | None = None) -> None:
        """Guarda los resultados en formato Parquet."""
        out_dir = Path(output_dir) if output_dir else self.processed_root
        out_dir.mkdir(parents=True, exist_ok=True)

        all_queries = []
        all_configs = []
        all_responses = []

        for name, (queries, configs, responses) in results.items():
            print(f"\nSaving {name}:")
            print(f"  queries: {len(queries)} rows")
            print(f"  configs: {len(configs)} rows")
            print(f"  responses: {len(responses)} rows")

            # Add source tag
            queries["source_dataset"] = name
            configs["source_dataset"] = name
            responses["source_dataset"] = name

            # Save
            queries_path = out_dir / f"queries_{name}.parquet"
            configs_path = out_dir / f"configs_{name}.parquet"
            responses_path = out_dir / f"responses_{name}.parquet"

            queries.to_parquet(queries_path, index=False)
            configs.to_parquet(configs_path, index=False)
            responses.to_parquet(responses_path, index=False)

            print(f"  Saved: {queries_path}")
            print(f"  Saved: {configs_path}")
            print(f"  Saved: {responses_path}")

            # Accumulate
            all_queries.append(queries)
            all_configs.append(configs)
            all_responses.append(responses)

        # Save combined
        if all_queries:
            combined_queries = pd.concat(all_queries, ignore_index=True)
            combined_configs = pd.concat(all_configs, ignore_index=True)
            combined_responses = pd.concat(all_responses, ignore_index=True)

            # Save as separate files with a manifest
            combined_queries.to_parquet(out_dir / "queries_combined.parquet", index=False)
            combined_configs.to_parquet(out_dir / "configs_combined.parquet", index=False)
            combined_responses.to_parquet(out_dir / "responses_combined.parquet", index=False)

            # Write manifest
            manifest = {
                "created_at": int(time.time()),
                "datasets": list(results.keys()),
                "total_queries": len(combined_queries),
                "total_configs": len(combined_configs),
                "total_responses": len(combined_responses),
                "files": {
                    "queries": "queries_combined.parquet",
                    "configs": "configs_combined.parquet",
                    "responses": "responses_combined.parquet",
                },
            }
            with open(out_dir / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

            print(f"\n{'=' * 60}")
            print(f"COMBINED DATASET:")
            print(f"{'=' * 60}")
            print(f"  Total queries: {len(combined_queries):,}")
            print(f"  Total configs: {len(combined_configs):,}")
            print(f"  Total responses: {len(combined_responses):,}")
            print(f"  Output: {out_dir}")

    def run(self, output_dir: str | None = None) -> None:
        """Ejecuta el pipeline completo."""
        print("=" * 60)
        print("ROUTEDATA PIPELINE")
        print("=" * 60)
        print(f"Raw data: {self.raw_root}")
        print(f"Output: {output_dir or self.processed_root}")

        start = time.time()

        results = self.load_all()

        if not results:
            print("\nERROR: No datasets were loaded successfully.")
            sys.exit(1)

        self.save(results, output_dir)

        elapsed = time.time() - start
        print(f"\nPipeline completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Load all datasets and generate unified schema")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("load", help="Load all datasets")

    args = parser.parse_args()

    if args.cmd == "load":
        pipeline = Pipeline()
        pipeline.run()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
