"""Juez LLM (conjunto) para tareas abiertas + muestreo para anotación humana.

Con solo pesos abiertos: usa ≥3 jueces de familias distintas y, si es posible, que NINGUNO
sea también un modelo del pool evaluado (sesgo de autopreferencia). Valida SIEMPRE contra
una muestra humana (objetivo: Pearson ≥ 0,8 y κ ≥ 0,7 tras binarizar), como hace R2-Bench
(500 respuestas, 30 anotadores; Pearson 0,82 con el juez elegido)."""

from __future__ import annotations

import json
import re

import numpy as np

from routedata.stats import cohen_kappa

from .backends import GenRequest

JUDGE_PROMPT = """You are a strict grader. Given a QUESTION, a REFERENCE answer and a RESPONSE, \
score the response from 0 (wrong/unhelpful) to 1 (fully correct and helpful). \
Reply ONLY with JSON: {{"score": <float 0-1>}}.

QUESTION:
{question}

REFERENCE:
{reference}

RESPONSE:
{response}
"""


def parse_score(text: str):
    m = re.search(r"\{[^{}]*\}", text, flags=re.S)
    try:
        return float(np.clip(json.loads(m.group(0))["score"], 0, 1)) if m else None
    except Exception:
        return None


def judge_ensemble(judge_backends, items, seed: int = 0):
    """items: lista de dicts con question/reference/response. Devuelve (media, desviación) por ítem."""
    all_scores = []
    for jb in judge_backends:
        reqs = [
            GenRequest(
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(**it)}],
                max_tokens=64,
                temperature=0.0,
                seed=seed,
            )
            for it in items
        ]
        all_scores.append([parse_score(r.text) for r in jb.generate(reqs)])
    A = np.array([[np.nan if s is None else s for s in col] for col in all_scores], float)
    return np.nanmean(A, axis=0), np.nanstd(A, axis=0)


def sample_for_human(df, n: int = 500, strata=("domain", "config_id"), seed: int = 0):
    """Muestra estratificada para anotación humana (≥2 anotadores por ítem recomendados)."""
    rng = np.random.default_rng(seed)
    groups = list(df.groupby(list(strata)).indices.values())
    per = max(1, n // max(1, len(groups)))
    idx: list[int] = []
    for g in groups:
        idx.extend(rng.choice(g, size=min(per, len(g)), replace=False))
    return df.iloc[sorted(idx)[:n]]


def judge_vs_human(judge, human, thr: float = 0.5):
    j, h = np.asarray(judge, float), np.asarray(human, float)
    pear = float(np.corrcoef(j, h)[0, 1]) if j.std() > 0 and h.std() > 0 else float("nan")
    return {"pearson": pear, "kappa": float(cohen_kappa(j >= thr, h >= thr)), "n": int(len(j))}
