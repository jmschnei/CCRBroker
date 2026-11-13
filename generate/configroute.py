"""ConfigRoute: respuestas por configuración (modelo × esfuerzo × presupuesto de longitud).

Flujo (modelo-a-modelo para caber en GPU; reanudable):
  1) sample_queries()  → muestra estratificada por dominio/dificultad
  2) build_config_grid() → tabla `configs`
  3) run()  → para cada modelo: carga backend, recorre sus configs, genera, etiqueta, guarda shards
Etiquetado: verificadores programáticos; las tareas abiertas quedan con correct=NaN y
label_source='pending_judge' hasta pasar por judge.py.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from routedata.schema import save_table

from .backends import GenRequest
from .efforts import EFFORTS, run_with_effort
from .prompts import build_messages
from .verifiers import verify


def sample_queries(sources: dict, quotas: dict, strat=("domain", "difficulty"), seed: int = 0):
    """sources: {nombre: DataFrame de queries}; quotas: {nombre: n}. Muestreo estratificado
    proporcional por `strat` (compatible con pandas>=3: sin groupby.apply)."""
    rng = np.random.default_rng(seed)
    parts = []
    for name, df in sources.items():
        n = min(quotas.get(name, len(df)), len(df))
        cols = [c for c in strat if c in df.columns]
        if not cols:
            parts.append(df.sample(n=n, random_state=seed))
            continue
        frac, picked = n / len(df), []
        for _, g in df.groupby(cols):
            k = min(len(g), max(1, int(round(len(g) * frac))))
            picked.append(g.sample(n=k, random_state=int(rng.integers(1 << 30))))
        parts.append(pd.concat(picked).head(n))
    return pd.concat(parts, ignore_index=True)


def build_config_grid(models: dict, budgets=(None,), efforts=("nothink",), hardware="", price_date=""):
    """models: {model: {'revision':..., 'price_in':..., 'price_out':..., 'thinking': bool, 'price_source':...}}.
    Solo los modelos con `thinking=True` reciben los niveles think_*; el resto solo 'nothink'."""
    rows = []
    for m, spec in models.items():
        effs = efforts if spec.get("thinking") else ["nothink"]
        # El presupuesto de longitud solo se cruza con 'nothink'; los niveles think_* ya acotan el
        # razonamiento por su propio presupuesto (cruzarlos duplicaría configuraciones idénticas).
        pairs = [(e, b) for e in effs for b in (budgets if e == "nothink" else (None,))]
        for e, b in pairs:
            cid = f"{m}|{e}|{'def' if b is None else b}"
            rows.append(
                dict(
                    config_id=cid,
                    model=m,
                    revision=spec["revision"],
                    effort=e,
                    budget=np.nan if b is None else float(b),
                    price_in=spec["price_in"],
                    price_out=spec["price_out"],
                    price_source=spec.get("price_source", "selfhost"),
                    price_date=price_date,
                    hardware=hardware,
                )
            )
    return pd.DataFrame(rows)


def _done_keys(out_dir):
    d = set()
    for f in os.listdir(out_dir) if os.path.isdir(out_dir) else []:
        if f.startswith("responses_") and f.endswith(".jsonl"):
            for line in open(os.path.join(out_dir, f)):
                r = json.loads(line)
                d.add((r["query_id"], r["config_id"], r["run_id"]))
    return d


def run(
    queries: pd.DataFrame,
    configs: pd.DataFrame,
    backend_factory,
    out_dir: str,
    run_id: str = "run0",
    snapshot_id: str = "s0",
    batch: int = 64,
    seed: int = 0,
    log=print,
):
    """backend_factory(model, revision) -> Backend. Reanuda saltando (query, config, run) ya hechos."""
    os.makedirs(out_dir, exist_ok=True)
    save_table(queries, os.path.join(out_dir, "queries"))
    save_table(configs, os.path.join(out_dir, "configs"))
    done = _done_keys(out_dir)
    for model, cfgs in configs.groupby("model", sort=False):
        todo_any = any((q, c, run_id) not in done for c in cfgs["config_id"] for q in queries["query_id"])
        if not todo_any:
            log(f"[skip] {model}")
            continue
        backend = backend_factory(model, cfgs["revision"].iloc[0])
        shard = os.path.join(out_dir, f"responses_{model.replace('/', '_')}.jsonl")
        with open(shard, "a") as fh:
            for _, cfg in cfgs.iterrows():
                eff = EFFORTS[cfg["effort"]]
                rows = [
                    r for _, r in queries.iterrows() if (r["query_id"], cfg["config_id"], run_id) not in done
                ]
                for i in range(0, len(rows), batch):
                    chunk = rows[i : i + batch]
                    reqs = []
                    for r in chunk:
                        b = None if np.isnan(cfg["budget"]) else int(cfg["budget"])
                        msgs, ttype = build_messages(r, budget=b)
                        meta = (
                            json.loads(r["meta"])
                            if isinstance(r.get("meta"), str) and r.get("meta")
                            else dict(r.get("meta") or {})
                        )
                        meta.update(query_id=r["query_id"], answer=r.get("answer", ""))
                        mt = (
                            (b if b else eff.answer_reserve)
                            if not eff.enable_thinking
                            else eff.answer_reserve
                        )
                        reqs.append(GenRequest(messages=msgs, max_tokens=mt, seed=seed, meta=meta))
                    _t0 = time.time()
                    res = run_with_effort(backend, reqs, eff)
                    for r, g in zip(chunk, res, strict=False):
                        _, ttype = build_messages(r)
                        _m = r.get("meta")
                        _m = json.loads(_m) if isinstance(_m, str) and _m else (_m or {})
                        corr, src = verify(ttype, g.text, r.get("answer", ""), tests=_m.get("tests"))
                        fh.write(
                            json.dumps(
                                dict(
                                    query_id=r["query_id"],
                                    config_id=cfg["config_id"],
                                    run_id=run_id,
                                    snapshot_id=snapshot_id,
                                    response_text=g.text,
                                    in_tokens=g.in_tokens,
                                    out_tokens=g.out_tokens,
                                    reasoning_tokens=g.reasoning_tokens,
                                    latency_s=g.latency_s,
                                    correct=float("nan") if corr is None else corr,
                                    label_source=src or "pending_judge",
                                    finish_reason=g.finish_reason,
                                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                                ),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    fh.flush()
                log(f"[ok] {cfg['config_id']} ({len(rows)} nuevas)")
        del backend
    return load_responses(out_dir)


def load_responses(out_dir):
    dfs = [
        pd.read_json(os.path.join(out_dir, f), lines=True)
        for f in sorted(os.listdir(out_dir))
        if f.startswith("responses_") and f.endswith(".jsonl")
    ]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
