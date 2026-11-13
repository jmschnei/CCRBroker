"""Carga el dataset unificado (data/processed/unified/) y muestra estadísticas + integridad.

Requiere haber ejecutado antes:  python -m routedata.pipeline load
Uso:  python examples/load_unified_dataset.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

UNIFIED = Path(__file__).resolve().parents[1] / "data" / "processed" / "unified"


def main() -> None:
    manifest = json.loads((UNIFIED / "manifest.json").read_text())
    print(
        f"Manifest: datasets={manifest['datasets']} | "
        f"{manifest['total_queries']:,} queries | "
        f"{manifest['total_configs']:,} configs | "
        f"{manifest['total_responses']:,} responses"
    )

    queries = pd.read_parquet(UNIFIED / "queries_combined.parquet")
    configs = pd.read_parquet(UNIFIED / "configs_combined.parquet")
    responses = pd.read_parquet(UNIFIED / "responses_combined.parquet", columns=["query_id", "config_id"])

    print("\nQueries por dataset:")
    print(queries["source_dataset"].value_counts().to_string())

    bad_queries = int((~responses["query_id"].isin(set(queries["query_id"]))).sum())
    bad_configs = int((~responses["config_id"].isin(set(configs["config_id"]))).sum())
    print(f"\nIntegridad referencial: {bad_queries} query_ids huérfanos, {bad_configs} config_ids huérfanos")
    assert bad_queries == 0 and bad_configs == 0, "La integridad referencial está rota"
    print("OK")


if __name__ == "__main__":
    main()
