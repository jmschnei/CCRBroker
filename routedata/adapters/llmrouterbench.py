"""LLMRouterBench → esquema unificado.

Entrada: data/raw/llmrouterbench/extracted/bench-release/<dataset>/<split>/<modelo>/*.json.
Cada JSON trae agregados (performance, cost, tokens...) y `records`, uno por consulta, con
index/instance_id, origin_query, prompt (plantilla de inferencia), prompt_tokens,
completion_tokens, cost, score, prediction y ground_truth.

Salida (una respuesta por par consulta-modelo OBSERVADO; no se rellena el producto cartesiano):
- queries: query_id = "<dataset>/<split>/<instance_id|index>", text = origin_query,
  prompt = prompt de inferencia, prompt_variant_count = nº de prompts distintos entre modelos,
  answer = ground_truth.
- responses: correct = score del record (label_source="per_record"); si falta, se usa el
  `performance` del fichero (label_source="aggregate_fallback": NO es etiqueta por instancia).
  Tokens negativos publicados se normalizan a ausentes (NaN).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = "llmrouterbench"
RESPONSE_TEXT_MAX = 2000  # la predicción se trunca: el texto completo sigue en data/raw


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


class LLMRouterBenchAdapter:
    """Adaptador de LLMRouterBench (records por instancia)."""

    def load(self, path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(f"LLMRouterBench root not found: {root}")
        json_files = sorted(root.rglob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {root}")
        print(f"Processing {len(json_files)} JSON files...")

        queries: dict[str, dict] = {}
        prompts: dict[str, set[str]] = {}
        rows: list[dict] = []
        for i, json_file in enumerate(json_files):
            if (i + 1) % 100 == 0:
                print(f"  Read {i + 1}/{len(json_files)} files...")
            try:
                with open(json_file) as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  Warning: could not read {json_file}: {e}")
                continue

            model = data.get("model_name", "unknown")
            dataset = data.get("dataset_name", "unknown")
            split = data.get("split", "unknown")
            aggregate = _num(data.get("performance"))
            for rec in data.get("records", []):
                rid = rec.get("instance_id") if rec.get("instance_id") not in (None, "") else rec.get("index")
                qid = f"{dataset}/{split}/{rid}"
                prompt = str(rec.get("prompt") or "")
                if qid not in queries:
                    queries[qid] = {
                        "query_id": qid,
                        "text": str(rec.get("origin_query") or rec.get("prompt") or ""),
                        "domain": f"{dataset}/{split}",
                        "answer": "" if rec.get("ground_truth") is None else str(rec.get("ground_truth")),
                        "prompt": prompt,
                    }
                    prompts[qid] = set()
                prompts[qid].add(prompt)

                score = _num(rec.get("score"))
                per_record = not np.isnan(score)
                rows.append(
                    {
                        "query_id": qid,
                        "config_id": model,
                        "response_text": (
                            "" if rec.get("prediction") is None else str(rec.get("prediction"))
                        )[:RESPONSE_TEXT_MAX],
                        "in_tokens": _num(rec.get("prompt_tokens")),
                        "out_tokens": _num(rec.get("completion_tokens")),
                        "correct": score if per_record else aggregate,
                        "label_source": "per_record" if per_record else "aggregate_fallback",
                        "cost_usd": _num(rec.get("cost")),
                    }
                )
        if not rows:
            raise ValueError("No records found in LLMRouterBench data")

        q = pd.DataFrame(list(queries.values()))
        q["prompt_variant_count"] = q["query_id"].map(lambda k: len(prompts[k]))
        q["category"] = q["domain"]
        q["source_dataset"] = SOURCE
        q["eval_only"] = False
        q["language"] = "en"
        q = q[
            [
                "query_id",
                "text",
                "domain",
                "source_dataset",
                "eval_only",
                "answer",
                "category",
                "language",
                "prompt",
                "prompt_variant_count",
            ]
        ]

        r = pd.DataFrame(rows)
        for col in ("in_tokens", "out_tokens"):
            r.loc[r[col] < 0, col] = np.nan  # contadores negativos publicados → ausentes
        r["run_id"] = "legacy"
        r["snapshot_id"] = "legacy"
        r["reasoning_tokens"] = 0
        r["latency_s"] = np.nan
        r["finish_reason"] = ""
        r["timestamp"] = ""
        r = r[
            [
                "query_id",
                "config_id",
                "run_id",
                "snapshot_id",
                "response_text",
                "in_tokens",
                "out_tokens",
                "reasoning_tokens",
                "latency_s",
                "correct",
                "label_source",
                "finish_reason",
                "timestamp",
                "cost_usd",
            ]
        ]

        models = sorted(r["config_id"].unique())
        c = pd.DataFrame(
            {
                "config_id": models,
                "model": models,
                "revision": "",
                "effort": "default",
                "budget": np.nan,
                "price_in": 0.0,
                "price_out": 0.0,
                "price_source": f"{SOURCE} (coste final por respuesta en cost_usd)",
                "price_date": "",
                "hardware": "",
            }
        )
        print(f"  Found {len(q)} unique queries, {len(models)} models, {len(r)} responses")
        return q, c, r


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
