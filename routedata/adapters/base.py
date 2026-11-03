"""Adaptadores: convierten datasets de routing públicos al esquema unificado.

IMPORTANTE: los esquemas de archivos externos no se han podido inspeccionar desde este
entorno (sin red). Los adaptadores son *defensivos* (autodetección + mapeo por YAML) y
cada uno trae un comando `inspect` que imprime columnas para que verifiques el mapeo
antes de usarlo.
"""
from __future__ import annotations
import pandas as pd

class EvalOnlyError(RuntimeError):
    pass

def assert_trainable(queries: pd.DataFrame):
    """Falla si alguna consulta pertenece a un dataset de solo evaluación (p. ej. RouterArena)."""
    if "eval_only" in queries.columns and queries["eval_only"].fillna(False).astype(bool).any():
        raise EvalOnlyError("El conjunto contiene datos 'eval_only' (p. ej. RouterArena): "
                            "prohibido entrenar/ajustar routers con ellos.")

def wide_to_tables(df: pd.DataFrame, prompt_col: str, perf_cols: list[str],
                   cost_suffix: str | None = None, cost_cols: dict[str, str] | None = None,
                   domain_col: str | None = None, id_col: str | None = None,
                   source_name: str = "unknown", eval_only: bool = False,
                   answer_col: str | None = None):
    """Formato ancho (una fila por consulta; columnas por modelo) → (queries, configs, responses).

    perf_cols: columnas con la calidad por modelo. El coste por modelo se toma de
    cost_cols[modelo] o de f"{modelo}{cost_suffix}". Como solo hay coste final (no tokens),
    los precios por token quedan en 0 y se usa `cost_usd`.
    """
    n = len(df)
    qid = df[id_col].astype(str) if id_col else pd.Series([f"{source_name}-{i}" for i in range(n)])
    queries = pd.DataFrame({
        "query_id": qid.values, "text": df[prompt_col].astype(str).values,
        "domain": (df[domain_col].astype(str).values if domain_col else ["unknown"] * n),
        "source_dataset": source_name, "eval_only": eval_only,
        "answer": (df[answer_col].astype(str).values if answer_col else ""),
    })
    queries["category"] = queries["domain"]; queries["language"] = "en"
    configs = pd.DataFrame({
        "config_id": perf_cols, "model": perf_cols, "revision": "", "effort": "default",
        "budget": float("nan"), "price_in": 0.0, "price_out": 0.0,
        "price_source": f"{source_name} (coste final por respuesta)", "price_date": "", "hardware": "",
    })
    rows = []
    for m in perf_cols:
        ccol = (cost_cols or {}).get(m) or (f"{m}{cost_suffix}" if cost_suffix else None)
        cost = df[ccol].values if ccol and ccol in df.columns else float("nan")
        rows.append(pd.DataFrame({
            "query_id": queries["query_id"].values, "config_id": m, "run_id": "legacy",
            "snapshot_id": "legacy", "response_text": "", "in_tokens": 0, "out_tokens": 0,
            "reasoning_tokens": 0, "latency_s": float("nan"), "correct": df[m].astype(float).values,
            "label_source": "legacy", "finish_reason": "", "timestamp": "", "cost_usd": cost}))
    return queries, configs, pd.concat(rows, ignore_index=True)
