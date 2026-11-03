"""Control de riesgo conformal para CCR (router de configuraciones con garantías).

Política con umbral t: ordena las configs por coste medio (barata→cara); elige la primera
cuya probabilidad predicha de éxito P̂ ≥ t; si ninguna, escala a la más fiable (argmax P̂).
Pérdida por consulta L_i(t) = 1 − Q[i, elegida(t)]. Si L es (aprox.) no creciente en t,
conformal risk control (Angelopoulos et al., 2022) da E[L(t̂)] ≤ α bajo intercambiabilidad:

    t̂ = min{ t : n/(n+1)·R̂_n(t) + B/(n+1) ≤ α },  B = 1.

Nota: la monotonía de L en t NO está garantizada para cualquier predictor; se comprueba
empíricamente (`monotonicity_violation`) y debe reportarse (ver E7).
"""
from __future__ import annotations
import numpy as np

def choose(Phat, cost_order, t):
    """Elecciones para umbral t. Phat[n,k]; cost_order: índices de configs de barata a cara."""
    ok = Phat[:, cost_order] >= t
    first = np.argmax(ok, axis=1)
    has = ok.any(axis=1)
    ch = np.where(has, np.asarray(cost_order)[first], np.argmax(Phat, axis=1))
    return ch

def risk_at(Q, Phat, cost_order, t):
    ch = choose(Phat, cost_order, t)
    return float(np.mean(1 - Q[np.arange(len(ch)), ch]))

def crc_threshold(Q_cal, Phat_cal, cost_order, alpha: float, grid=None):
    """Umbral CRC más bajo (= más barato) que cumple el límite de riesgo. None si no existe."""
    n = len(Q_cal); grid = np.linspace(0, 1, 201) if grid is None else grid
    for t in grid:
        if n / (n + 1) * risk_at(Q_cal, Phat_cal, cost_order, t) + 1 / (n + 1) <= alpha:
            return float(t)
    return None

def monotonicity_violation(Q, Phat, cost_order, grid=None):
    """Mayor subida de riesgo al aumentar t (0 = perfectamente monótona)."""
    grid = np.linspace(0, 1, 201) if grid is None else grid
    r = np.array([risk_at(Q, Phat, cost_order, t) for t in grid])
    return float(np.max(np.maximum(np.diff(r), 0))) if len(r) > 1 else 0.0

def stratified_crc(Q_cal, Phat_cal, strata_cal, cost_order, alpha, grid=None):
    """Un umbral por estrato (dominio/dificultad). Estratos con n pequeño caen al global."""
    glob = crc_threshold(Q_cal, Phat_cal, cost_order, alpha, grid)
    out = {}
    for s in np.unique(strata_cal):
        m = strata_cal == s
        t = crc_threshold(Q_cal[m], Phat_cal[m], cost_order, alpha, grid) if m.sum() >= 30 else None
        out[s] = t if t is not None else glob
    return out, glob

def apply_stratified(Phat, strata, thresholds, cost_order, fallback):
    ch = np.zeros(len(Phat), int)
    for s in np.unique(strata):
        m = strata == s; t = thresholds.get(s, fallback)
        t = 1.0 if t is None else t  # sin umbral válido: escalar siempre
        ch[m] = choose(Phat[m], cost_order, t)
    return ch
