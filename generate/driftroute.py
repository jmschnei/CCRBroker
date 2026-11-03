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
import json, os, platform, time
from dataclasses import dataclass, asdict, field
import numpy as np, pandas as pd
from routedata.stats import mcnemar_exact

@dataclass
class SnapshotManifest:
    snapshot_id: str
    date: str
    models: dict                         # {model: revision_hash}
    engine: str = "vllm"
    engine_version: str = ""
    quantization: str = "none"
    hardware: str = ""
    price_table: dict = field(default_factory=dict)   # {model: {"in":..,"out":..,"source":..}}
    events: list = field(default_factory=list)        # [{"type":"add_model","model":...}, ...]
    notes: str = ""
    python: str = platform.python_version()
    def save(self, d):
        os.makedirs(d, exist_ok=True)
        json.dump(asdict(self), open(os.path.join(d, f"manifest_{self.snapshot_id}.json"), "w"), indent=2)

EVENT_TYPES = ("add_model", "retire_model", "new_revision", "price_change", "new_query_batch")

def drift_report(resp_a: pd.DataFrame, resp_b: pd.DataFrame, config_col: str = "config_id"):
    """Compara dos snapshots por config: Δ precisión, Δ tokens de salida y flips (McNemar)."""
    a = resp_a.set_index(["query_id", config_col]); b = resp_b.set_index(["query_id", config_col])
    common = a.index.intersection(b.index); rows = []
    for cfg in sorted({c for _, c in common}):
        idx = [i for i in common if i[1] == cfg]
        ca, cb = a.loc[idx, "correct"].values, b.loc[idx, "correct"].values
        p, n01, n10 = mcnemar_exact(ca, cb)
        rows.append(dict(config=cfg, n=len(idx), acc_a=float(np.mean(ca)), acc_b=float(np.mean(cb)),
                         delta_acc=float(np.mean(cb) - np.mean(ca)), flips_a_to_b_wrong=n10,
                         flips_a_to_b_right=n01, p_mcnemar=p,
                         delta_out_tokens=float(b.loc[idx, "out_tokens"].mean() - a.loc[idx, "out_tokens"].mean())))
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
