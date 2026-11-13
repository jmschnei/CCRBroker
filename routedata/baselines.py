"""Baselines obligatorios (triviales) y un router kNN de referencia para validar el pipeline."""

from __future__ import annotations

import numpy as np

from . import metrics as M


def trivial_baselines(Q, C, groups=None, seed: int = 0):
    """Devuelve {nombre: vector de elecciones}. 'best_single' y 'oracle' son a posteriori
    (cotas de referencia, no routers desplegables)."""
    n, k = Q.shape
    rng = np.random.default_rng(seed)
    out = {}
    out["cheapest"] = np.full(n, int(np.nanargmin(np.nanmean(C, axis=0))))
    out["best_single"] = np.full(n, M.best_single(Q, groups))
    out["random"] = rng.integers(0, k, size=n)
    out["oracle"] = M.oracle_choice(Q, C)
    if groups is not None:
        g = np.asarray(groups)
        ch = np.zeros(n, int)
        for u in np.unique(g):
            m = g == u
            ch[m] = int(np.nanargmax(np.nanmean(Q[m], axis=0)))
        out["best_single_per_group"] = ch
    return out


class KNNRouter:
    """Router de referencia: promedia Q y C de los k vecinos (coseno) en el espacio de
    embeddings. Sirve de baseline fuerte-y-simple y para probar el pipeline."""

    def __init__(self, k: int = 20):
        self.k = k

    def fit(self, X, Q, C):
        from sklearn.neighbors import NearestNeighbors

        self.nn = NearestNeighbors(n_neighbors=min(self.k, len(X)), metric="cosine").fit(X)
        self.Q, self.C = np.asarray(Q), np.asarray(C)
        return self

    def predict(self, X):
        _, idx = self.nn.kneighbors(X)
        return self.Q[idx].mean(axis=1), self.C[idx].mean(axis=1)
