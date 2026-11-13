"""Métricas de evaluación de routing sobre matrices Q (calidad) y C (coste).

Convención: Q[n,k] en [0,1] (1 = correcto), C[n,k] en USD; `choice[n]` = índice de
configuración elegida por el router para cada consulta. Las definiciones de
Gain@b, Gap@O, PerfGain y CostSave siguen LLMRouterBench; AUDC/QNC siguen la
formulación de deferral curve usada en R2-Router/UniRouter; AIQ es una
implementación propia aproximada de la de RouterBench (verificar contra el
código oficial antes de reportar números comparables).
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ básicos
def realized(Q, C, choice):
    idx = np.arange(len(choice))
    return Q[idx, choice], C[idx, choice]


def mean_quality_cost(Q, C, choice):
    q, c = realized(Q, C, choice)
    return float(np.mean(q)), float(np.mean(c))


def oracle_choice(Q, C, tau: float = 0.999):
    """Por consulta: la config más barata entre las 'correctas' (Q>=tau); si ninguna
    acierta, la más barata (no aporta calidad)."""
    n, k = Q.shape
    correct = Q >= tau
    cost_masked = np.where(correct, C, np.inf)
    ch = np.argmin(cost_masked, axis=1)
    none = ~correct.any(axis=1)
    ch[none] = np.argmin(C[none], axis=1)
    return ch


def best_single(Q, groups=None):
    """Índice del mejor modelo único *a posteriori* (media de las medias por grupo, como
    en LLMRouterBench, o media global si no hay grupos)."""
    if groups is None:
        return int(np.nanargmax(np.nanmean(Q, axis=0)))
    groups = np.asarray(groups)
    per = np.stack([np.nanmean(Q[groups == g], axis=0) for g in np.unique(groups)])
    return int(np.argmax(per.mean(axis=0)))


def _group_acc(q, groups):
    if groups is None:
        return {"_all": float(np.mean(q))}
    groups = np.asarray(groups)
    return {g: float(np.mean(q[groups == g])) for g in np.unique(groups)}


def gain_at(Q, C, choice, base_choice, groups=None):
    """Gain@b = media_d( Acc(a,d)/Acc(b,d) - 1 )."""
    a = _group_acc(Q[np.arange(len(choice)), choice], groups)
    b = _group_acc(Q[np.arange(len(choice)), base_choice], groups)
    return float(np.mean([a[g] / b[g] - 1 for g in a if b[g] > 0]))


def gap_at_oracle(Q, C, choice, groups=None, tau=0.999):
    """Gap@O = media_d( 1 - Acc(a,d)/Acc(O,d) )."""
    o = oracle_choice(Q, C, tau)
    a = _group_acc(Q[np.arange(len(choice)), choice], groups)
    b = _group_acc(Q[np.arange(len(o)), o], groups)
    return float(np.mean([1 - a[g] / b[g] for g in a if b[g] > 0]))


def recall_failure_rate(Q, choice, max_correct: int = 3, tau=0.999):
    """Accuracy del router en consultas 'difíciles' (<= max_correct configs aciertan)."""
    ncorr = (Q >= tau).sum(axis=1)
    hard = (ncorr >= 1) & (ncorr <= max_correct)
    if hard.sum() == 0:
        return float("nan"), 0
    return float(np.mean(Q[np.arange(len(choice)), choice][hard])), int(hard.sum())


# ------------------------------------------------------------------ Pareto
def pareto_frontier(points):
    """points: array (m,2) con (coste, calidad). Devuelve índices no dominados
    (menor coste, mayor calidad), ordenados por coste."""
    P = np.asarray(points, float)
    order = np.lexsort((-P[:, 1], P[:, 0]))
    best_q, keep = -np.inf, []
    for i in order:
        if P[i, 1] > best_q + 1e-12:
            keep.append(i)
            best_q = P[i, 1]
    return np.array(keep)


def hypervolume_2d(points, ref_cost: float, ref_quality: float = 0.0):
    """Área dominada respecto al punto de referencia (ref_cost alto, ref_quality bajo)."""
    P = np.asarray(points, float)
    fr = P[pareto_frontier(P)]
    fr = fr[fr[:, 0] <= ref_cost]
    area, prev_q = 0.0, ref_quality
    for c, q in fr:
        if q > prev_q:
            area += (ref_cost - c) * (q - prev_q)
            prev_q = q
    return float(area)


def pareto_dist(method_points, frontier_points):
    """Distancia L1 media a la frontera, con coordenadas normalizadas a [0,1] (ParetoDist)."""
    M, F = np.asarray(method_points, float), np.asarray(frontier_points, float)
    allp = np.vstack([M, F])
    lo, hi = allp.min(0), allp.max(0)
    sc = np.where(hi - lo > 0, hi - lo, 1.0)
    Mn, Fn = (M - lo) / sc, (F - lo) / sc
    return float(np.mean([np.min(np.abs(Fn - m).sum(axis=1)) for m in Mn]))


def perf_gain_cost_save(points, best_single_point):
    """points: [(coste, calidad)] de todas las configs de un router; best_single_point=(c_B, q_B).
    PerfGain = max_q/q_B - 1 ; CostSave = 1 - min{coste: q>=q_B}/c_B (None si no alcanza)."""
    P = np.asarray(points, float)
    cB, qB = best_single_point
    perf = float(P[:, 1].max() / qB - 1)
    ok = P[P[:, 1] >= qB - 1e-12]
    save = float(1 - ok[:, 0].min() / cB) if len(ok) else None
    return perf, save


def aiq(points, c_min=None, c_max=None):
    """Área normalizada bajo la envolvente cóncava no decreciente de (coste, calidad).
    Aproximación de la AIQ de RouterBench (verificar contra la implementación oficial)."""
    P = np.asarray(points, float)
    fr = P[pareto_frontier(P)]
    c_min = fr[0, 0] if c_min is None else c_min
    c_max = fr[-1, 0] if c_max is None else c_max
    if c_max <= c_min:
        return float(fr[:, 1].max())
    # envolvente cóncava superior
    hull = []
    for p in fr:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(tuple(p))
    H = np.array(hull)
    xs = np.linspace(c_min, c_max, 512)
    ys = np.interp(xs, H[:, 0], H[:, 1], left=H[0, 1], right=H[-1, 1])
    return float(np.trapezoid(ys, xs) / (c_max - c_min))


# ------------------------------------------------------------------ deferral curve
def deferral_curve(Qhat, Chat, Q, C, lambdas=None):
    """Barre λ en [0,1]; el router elige argmax (1-λ)·Q̂ − λ·Ĉn (Ĉ normalizado por su máximo).
    Devuelve (coste_medio_real, calidad_media_real) por λ."""
    lambdas = np.linspace(0, 1, 101) if lambdas is None else lambdas
    scale = np.nanmax(Chat) if np.nanmax(Chat) > 0 else 1.0
    pts = []
    for lam in lambdas:
        ch = np.argmax((1 - lam) * Qhat - lam * (Chat / scale), axis=1)
        q, c = mean_quality_cost(Q, C, ch)
        pts.append((c, q))
    return np.array(pts)


def audc(curve, cost_norm: float):
    """Área bajo la deferral curve con coste normalizado por `cost_norm`
    (p. ej. coste medio de la config más cara). La curva se hace monótona."""
    P = curve[np.argsort(curve[:, 0])]
    x = np.clip(P[:, 0] / cost_norm, 0, 1)
    y = np.maximum.accumulate(P[:, 1])
    x = np.concatenate([[0.0], x, [1.0]])
    y = np.concatenate([[y[0]], y, [y[-1]]])
    return float(np.trapezoid(y, x))


def qnc(curve, best_model_quality: float, best_model_cost: float):
    """Query-Normalized Cost: coste mínimo relativo para igualar la calidad del mejor
    modelo (1.0 si no se alcanza)."""
    P = curve[curve[:, 1] >= best_model_quality - 1e-12]
    return float(min(1.0, P[:, 0].min() / best_model_cost)) if len(P) else 1.0


# ------------------------------------------------------------------ robustez / calibración
def consistency(choice_a, choice_b):
    """Fracción de consultas donde el router elige lo mismo ante la consulta original y
    la perturbada (consistencia ante perturbaciones)."""
    return float(np.mean(np.asarray(choice_a) == np.asarray(choice_b)))


def cost_inflation(C, choice_clean, choice_attacked):
    """Multiplicador de coste medio bajo ataque."""
    idx = np.arange(len(choice_clean))
    return float(np.mean(C[idx, choice_attacked]) / np.mean(C[idx, choice_clean]))


def ece(p, y, bins: int = 10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def risk_coverage(conf, y):
    """Curva riesgo-cobertura: ordena por confianza descendente."""
    o = np.argsort(-np.asarray(conf))
    err = 1 - np.asarray(y, float)[o]
    cov = np.arange(1, len(o) + 1) / len(o)
    return cov, np.cumsum(err) / np.arange(1, len(o) + 1)
