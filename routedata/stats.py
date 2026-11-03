"""Estadística para comparar routers: bootstrap por consulta, pruebas pareadas y
corrección por comparaciones múltiples."""
from __future__ import annotations
from math import comb
import numpy as np

def bootstrap_ci(values, stat=np.mean, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    v = np.asarray(values, float); rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    s = np.array([stat(v[i]) for i in idx]) if stat is not np.mean else v[idx].mean(axis=1)
    return float(stat(v)), float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2))

def paired_bootstrap_diff(a, b, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """IC de mean(a-b) remuestreando consultas (a y b alineados)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    m, lo, hi = bootstrap_ci(d, n_boot=n_boot, alpha=alpha, seed=seed)
    return m, lo, hi

def mcnemar_exact(correct_a, correct_b):
    """Prueba exacta de McNemar (binomial bilateral) sobre aciertos binarios pareados."""
    a, b = np.asarray(correct_a) >= 0.5, np.asarray(correct_b) >= 0.5
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum()); n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return float(p), n01, n10

def holm(pvals, alpha: float = 0.05):
    """Corrección de Holm. Devuelve (p_ajustados, rechazos)."""
    p = np.asarray(pvals, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj, adj <= alpha

def benjamini_hochberg(pvals, alpha: float = 0.05):
    p = np.asarray(pvals, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); prev = 1.0
    for r in range(m - 1, -1, -1):
        i = o[r]; prev = min(prev, p[i] * m / (r + 1)); adj[i] = prev
    return adj, adj <= alpha

def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b); labs = np.unique(np.concatenate([a, b]))
    po = float(np.mean(a == b))
    pe = float(sum(np.mean(a == l) * np.mean(b == l) for l in labs))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0
