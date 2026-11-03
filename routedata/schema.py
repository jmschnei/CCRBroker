"""Esquema unificado de datos para routing.

Tres tablas (formato largo, una fila por entidad):
  queries   : una fila por consulta
  configs   : una fila por configuración (modelo, esfuerzo, presupuesto)
  responses : una fila por (query_id, config_id, run_id)

Se guardan en parquet si pyarrow está disponible; si no, en JSONL.
"""
from __future__ import annotations
import json, os
from typing import Iterable
import numpy as np
import pandas as pd

QUERY_COLS = {
    "query_id": "str", "text": "str", "options": "json", "answer": "str",
    "domain": "str", "category": "str", "source_dataset": "str",
    "difficulty": "str", "language": "str", "cluster_id": "int",
    "created_at": "str", "eval_only": "bool", "meta": "json",
}
CONFIG_COLS = {
    "config_id": "str", "model": "str", "revision": "str", "effort": "str",
    "budget": "float", "price_in": "float", "price_out": "float",
    "price_source": "str", "price_date": "str", "hardware": "str",
}
RESPONSE_COLS = {
    "query_id": "str", "config_id": "str", "run_id": "str", "snapshot_id": "str",
    "response_text": "str", "in_tokens": "int", "out_tokens": "int",
    "reasoning_tokens": "int", "latency_s": "float", "correct": "float",
    "label_source": "str", "finish_reason": "str", "timestamp": "str",
    "cost_usd": "float (opcional; datasets heredados que solo traen coste)",
}
_REQUIRED = {
    "queries": ["query_id", "text", "domain", "source_dataset"],
    "configs": ["config_id", "model", "price_in", "price_out"],
    "responses": ["query_id", "config_id", "in_tokens", "out_tokens", "correct"],
}

def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False

def save_table(df: pd.DataFrame, path_noext: str) -> str:
    """Guarda en parquet (si hay pyarrow) o JSONL. Devuelve la ruta escrita."""
    os.makedirs(os.path.dirname(path_noext) or ".", exist_ok=True)
    if _has_parquet():
        p = path_noext + ".parquet"; df.to_parquet(p, index=False)
    else:
        p = path_noext + ".jsonl"; df.to_json(p, orient="records", lines=True, force_ascii=False)
    return p

def load_table(path_noext: str) -> pd.DataFrame:
    for ext, fn in ((".parquet", pd.read_parquet),
                    (".jsonl", lambda p: pd.read_json(p, lines=True))):
        if os.path.exists(path_noext + ext):
            return fn(path_noext + ext)
    raise FileNotFoundError(path_noext)

def validate(df: pd.DataFrame, kind: str) -> list[str]:
    """Devuelve lista de problemas (vacía si todo OK)."""
    problems = []
    for c in _REQUIRED[kind]:
        if c not in df.columns:
            problems.append(f"{kind}: falta columna {c}")
    if problems:
        return problems
    if kind == "queries" and df["query_id"].duplicated().any():
        problems.append("queries: query_id duplicado")
    if kind == "configs" and df["config_id"].duplicated().any():
        problems.append("configs: config_id duplicado")
    if kind == "responses":
        key = ["query_id", "config_id"] + (["run_id"] if "run_id" in df.columns else [])
        if df.duplicated(key).any():
            problems.append("responses: pares (query, config, run) duplicados")
        if not df["correct"].between(0, 1).all():
            problems.append("responses: 'correct' fuera de [0,1]")
        if (df["in_tokens"] < 0).any() or (df["out_tokens"] < 0).any():
            problems.append("responses: tokens negativos")
    return problems

def to_matrices(queries: pd.DataFrame, configs: pd.DataFrame, responses: pd.DataFrame,
                run_id: str | None = None):
    """Convierte tablas largas en matrices Q[n,k] (calidad) y C[n,k] (coste USD).

    Las celdas sin respuesta quedan como NaN; usa `mask` para filtrar.
    Coste = (in_tokens*price_in + (out_tokens)*price_out)/1e6, donde out_tokens
    ya incluye los tokens de razonamiento (se facturan como salida).
    """
    r = responses if run_id is None else responses[responses["run_id"] == run_id]
    r = r.merge(configs[["config_id", "price_in", "price_out"]], on="config_id")
    r = r.assign(cost=(r["in_tokens"] * r["price_in"] + r["out_tokens"] * r["price_out"]) / 1e6)
    if "cost_usd" in r.columns:  # datasets heredados que solo traen coste final
        r["cost"] = r["cost_usd"].where(r["cost_usd"].notna(), r["cost"])
    qi = {q: i for i, q in enumerate(queries["query_id"])}
    ci = {c: j for j, c in enumerate(configs["config_id"])}
    Q = np.full((len(qi), len(ci)), np.nan); C = Q.copy(); T = Q.copy()
    for q, c, y, cost, tok in zip(r["query_id"], r["config_id"], r["correct"], r["cost"], r["out_tokens"]):
        if q in qi and c in ci:
            Q[qi[q], ci[c]] = y; C[qi[q], ci[c]] = cost; T[qi[q], ci[c]] = tok
    return Q, C, T
