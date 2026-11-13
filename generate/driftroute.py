"""DriftRoute: deriva temporal con modelos de pesos abiertos.

Qué *sí* puede derivar con pesos abiertos y pin de revisión (fuentes de deriva):
  (a) nuevas versiones/lanzamientos de modelos (cambio de `revision` o de familia),
  (b) cambios de cuantización, motor de inferencia (versión de vLLM), hardware o semilla,
  (c) cambios de precio (alquiler de GPU, precios de proveedores de los mismos pesos),
  (d) altas y bajas de modelos en el pool,
  (e) deriva de *consultas*: lotes nuevos posteriores al corte (p. ej. problemas de concurso).
Qué NO hay con pesos abiertos: cambios silenciosos de un endpoint propietario. Limitación a
declarar en el artículo (el benchmark mide deriva de pool, no de API cerrada).
Cada snapshot guarda un manifiesto para reproducir y para auditar qué cambió.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as _md
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from routedata.stats import mcnemar_exact


@dataclass
class SnapshotManifest:
    snapshot_id: str
    date: str
    models: dict  # {model: revision_hash}
    engine: str = "vllm"
    engine_version: str = ""
    quantization: str = "none"
    hardware: str = ""
    price_table: dict = field(default_factory=dict)  # {model: {"in":..,"out":..,"source":..}}
    events: list = field(default_factory=list)  # [{"type":"add_model","model":...}, ...]
    notes: str = ""
    python: str = platform.python_version()
    # --- campos para auditar COMPARABILIDAD entre snapshots (deben coincidir salvo lo que el evento cambie)
    queries_hash: str = ""  # hash de los query_id del subconjunto congelado (+ lotes)
    prompt_hash: str = ""  # hash de plantillas de prompt y niveles de esfuerzo
    gen_params: dict = field(default_factory=dict)  # temperature, top_p, seed
    judge: dict = field(default_factory=dict)  # {"models": {m: revision}, "prompt_hash": ...}
    models_meta: dict = field(default_factory=dict)  # {model: {"size_b":..., "thinking":...}}
    grid: dict = field(default_factory=dict)  # efforts, budgets

    def save(self, d):
        os.makedirs(d, exist_ok=True)
        json.dump(asdict(self), open(os.path.join(d, f"manifest_{self.snapshot_id}.json"), "w"), indent=2)


EVENT_TYPES = ("add_model", "retire_model", "new_revision", "price_change", "new_query_batch")


def drift_report(resp_a: pd.DataFrame, resp_b: pd.DataFrame, config_col: str = "config_id"):
    """Compara dos snapshots por config: Δ precisión, Δ tokens de salida y flips (McNemar)."""
    a = resp_a.set_index(["query_id", config_col])
    b = resp_b.set_index(["query_id", config_col])
    common = a.index.intersection(b.index)
    rows = []
    for cfg in sorted({c for _, c in common}):
        idx = [i for i in common if i[1] == cfg]
        ca, cb = a.loc[idx, "correct"].values, b.loc[idx, "correct"].values
        p, n01, n10 = mcnemar_exact(ca, cb)
        rows.append(
            dict(
                config=cfg,
                n=len(idx),
                acc_a=float(np.mean(ca)),
                acc_b=float(np.mean(cb)),
                delta_acc=float(np.mean(cb) - np.mean(ca)),
                flips_a_to_b_wrong=n10,
                flips_a_to_b_right=n01,
                p_mcnemar=p,
                delta_out_tokens=float(b.loc[idx, "out_tokens"].mean() - a.loc[idx, "out_tokens"].mean()),
            )
        )
    return pd.DataFrame(rows)


def apply_events(configs: pd.DataFrame, events: list) -> pd.DataFrame:
    """Aplica eventos de pool/precio a una tabla `configs` (para simulaciones y replays)."""
    c = configs.copy()
    for e in events:
        if e["type"] == "retire_model":
            c = c[c["model"] != e["model"]]
        elif e["type"] == "price_change":
            m = c["model"] == e["model"]
            c.loc[m, "price_in"] *= e.get("factor_in", e.get("factor", 1.0))
            c.loc[m, "price_out"] *= e.get("factor_out", e.get("factor", 1.0))
        elif e["type"] == "add_model":
            c = pd.concat([c, pd.DataFrame(e["configs"])], ignore_index=True)
    return c.reset_index(drop=True)


# ------------------------------------------------------------------ utilidades de Fase 3


def capture_environment() -> dict:
    """Versiones relevantes de la pila de inferencia y GPU (para atribuir deriva a motor/hardware)."""
    env: dict[str, Any] = {}
    for pkg in ("vllm", "torch", "transformers", "flash-attn", "xformers"):
        try:
            env[pkg] = _md.version(pkg)
        except Exception:
            pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            env["gpus"] = [l.strip() for l in out.stdout.strip().splitlines()]
    except Exception:
        pass
    return env


def ids_hash(ids) -> str:
    return hashlib.md5("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def prompt_hash() -> str:
    from . import efforts as E
    from . import prompts as P

    blob = P.MC_TEMPLATE + P.MATH_TEMPLATE + P.OPEN_TEMPLATE + E.CLOSE_THINK + repr(sorted(E.EFFORTS.items()))
    return hashlib.md5(blob.encode()).hexdigest()


def comparability_check(manifests: list) -> list:
    """Devuelve avisos si los snapshots no son comparables (consultas, prompts, parámetros o juez
    distintos). Un aviso significa que la 'deriva' medida puede ser un artefacto del protocolo."""
    w = []
    base = manifests[0]
    for m in manifests[1:]:
        for fld in ("prompt_hash", "gen_params", "judge", "grid"):
            if getattr(m, fld) != getattr(base, fld):
                w.append(f"{m.snapshot_id}: '{fld}' difiere de {base.snapshot_id}")
        qh_b = base.queries_hash
        qh_m = m.queries_hash
        if qh_b and qh_m and qh_b != qh_m and not any(e.get("type") == "new_query_batch" for e in m.events):
            w.append(f"{m.snapshot_id}: conjunto de consultas distinto sin evento new_query_batch")
    return w


def freeze_subset(
    queries,
    n: int,
    seed: int = 0,
    verifiable_only: bool = False,
    exclude_ids=(),
    strat=("domain", "difficulty"),
):
    """Congela el subconjunto de DriftRoute: estratificado, sin la partición privada, y (opcional)
    solo con tareas verificables para no depender del juez de la Fase 2."""
    import json

    import numpy as np

    q = queries[~queries["query_id"].isin(set(exclude_ids))].copy()
    if verifiable_only:
        tt = q["meta"].apply(
            lambda m: (json.loads(m) if isinstance(m, str) and m else (m or {})).get("task_type", "open")
        )
        q = q[tt.isin(["mc", "math", "code"])]
    q = q.drop_duplicates("query_id")
    cols = [c for c in strat if c in q.columns]
    rng = np.random.default_rng(seed)
    frac = min(1.0, n / max(1, len(q)))
    parts = []
    for _, g in q.groupby(cols) if cols else [(0, q)]:
        parts.append(
            g.sample(
                n=min(len(g), max(1, int(round(len(g) * frac)))), random_state=int(rng.integers(1 << 30))
            )
        )
    out = __import__("pandas").concat(parts).head(n).reset_index(drop=True)
    out["batch_id"] = "b0"
    return out


def split_train_test(queries, test_frac: float = 0.5, seed: int = 0):
    """Partición FIJA por cluster (sin casi-duplicados repartidos). No cambiar entre snapshots."""
    from routedata.splits import iid_split

    g = queries["cluster_id"].values if "cluster_id" in queries.columns else None
    tr, te = iid_split(len(queries), test_frac, seed, groups=g)
    return list(queries["query_id"].iloc[tr]), list(queries["query_id"].iloc[te])


def apply_pool_events(
    pool: dict, snapshot_id: str, add=(), retire=(), revision=(), price_factor=(), new_batch=False
):
    """Aplica eventos a un dict de pool (para `new-pool`). Devuelve (pool_nuevo, eventos)."""
    import copy

    p = copy.deepcopy(pool)
    ev = []
    for spec in add:  # "model=revision,size_b,thinking(0/1),tp"
        m, rest = spec.split("=", 1)
        rev, size, think, tp = rest.split(",")
        p["models"][m] = dict(revision=rev, thinking=bool(int(think)), size_b=float(size), tp=int(tp))
        ev.append(dict(type="add_model", model=m, snapshot=snapshot_id))
    for m in retire:
        p["models"].pop(m)
        ev.append(dict(type="retire_model", model=m, snapshot=snapshot_id))
    for spec in revision:
        m, rev = spec.split("=", 1)
        old = p["models"][m]["revision"]
        p["models"][m]["revision"] = rev
        ev.append(dict(type="new_revision", model=m, old=old, new=rev, snapshot=snapshot_id))
    for spec in price_factor:
        m, f = spec.split("=", 1)
        p["models"][m]["price_factor"] = float(f) * p["models"][m].get("price_factor", 1.0)
        ev.append(dict(type="price_change", model=m, factor=float(f), snapshot=snapshot_id))
    if new_batch:
        ev.append(dict(type="new_query_batch", snapshot=snapshot_id))
    p["events"] = ev
    return p, ev


def load_snaps(drift_dir: str, featurize=None, dim: int = 64):
    """Lee snapshots (s00, s01, ...) del disco y construye `routedata.drift.Snap` comparables.
    La featurización se calcula UNA vez sobre la unión de consultas (representación idéntica en todos)."""
    import json
    import os
    import re

    import numpy as np
    import pandas as pd

    from routedata.drift import Snap, featurize_tfidf, profile_vector
    from routedata.schema import load_table, to_matrices

    from . import configroute as CR
    from . import phase2 as P2

    sids = sorted(
        d
        for d in os.listdir(drift_dir)
        if re.fullmatch(r"s\d+", d) and os.path.isdir(os.path.join(drift_dir, d))
    )
    raw = []
    for sid in sids:
        d = os.path.join(drift_dir, sid)
        q = load_table(os.path.join(d, "queries"))
        c = load_table(os.path.join(d, "configs"))
        r = CR.load_responses(d)
        for f in ("labels_judge.jsonl", "labels_human.jsonl"):
            pth = os.path.join(d, f)
            if os.path.exists(pth):
                r = P2.overlay_labels(r, pd.read_json(pth, lines=True))
        man = json.load(open(os.path.join(d, f"manifest_{sid}.json")))
        raw.append((sid, q, c, r, man))
    allq = (
        pd.concat([x[1][["query_id", "text"]] for x in raw])
        .drop_duplicates("query_id")
        .reset_index(drop=True)
    )
    X_all = (featurize or featurize_tfidf)(allq["text"].tolist())
    pos = {q: i for i, q in enumerate(allq["query_id"])}
    snaps = []
    for sid, q, c, r, man in raw:
        Q, C, _ = to_matrices(q, c, r)
        meta = man.get("models_meta", {})
        prof = np.stack(
            [
                profile_vector(
                    meta.get(m, {}).get("size_b", 8),
                    meta.get(m, {}).get("thinking", False),
                    e,
                    b if b == b else None,
                    po,
                )
                for m, e, b, po in zip(c["model"], c["effort"], c["budget"], c["price_out"], strict=False)
            ]
        )
        batch = q["batch_id"].values if "batch_id" in q.columns else np.array(["b0"] * len(q))
        snaps.append(
            Snap(
                sid,
                list(c["config_id"]),
                X_all[[pos[x] for x in q["query_id"]]],
                q["query_id"].values,
                Q,
                C,
                c["price_out"].values.astype(float),
                prof,
                batch,
                man.get("events", []),
            )
        )
    return snaps
