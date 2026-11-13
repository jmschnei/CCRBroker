"""E4 (RQ4 · H4): routers frente a deriva del pool y evaluación *prequential* por snapshots.

Protocolo (evita fugas):
  - Las consultas del subconjunto DriftRoute se dividen UNA vez por cluster en `train_ids` y `test_ids`
    y la división no cambia entre snapshots. Los lotes nuevos (deriva de consultas) van solo a test.
  - En cada snapshot s, el router puede adaptarse con una fracción f de las consultas de train_ids
    *con feedback real en s* ('tráfico de calibración', evaluado en sombra sobre todas las configs).
  - Se evalúa siempre sobre test_ids en s.
Routers: StaticRouter (entrenado en s0, nunca actualizado), RefitRouter(f) (kNN reentrenado en s con la
fracción f; f=1 es la cota superior de 'reentrenar'), ProfileOnlineRouter(f) (priores por perfil de
configuración para configs nuevas + corrección online de sesgo de calidad y escala de coste).
Métricas: AIQ relativa al techo (oráculo + configs individuales), pérdida relativa del estático frente
al reentrenado, y fracción de tráfico necesaria para recuperar el 80 % de la pérdida.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import metrics as M


@dataclass
class Snap:
    sid: str
    config_ids: list
    X: np.ndarray  # (n, d) rasgos de consulta
    qids: np.ndarray  # (n,)
    Q: np.ndarray  # (n, k) calidad (NaN si falta)
    C: np.ndarray  # (n, k) coste con los precios DE ESTE snapshot
    price: np.ndarray  # (k,) índice de precio (p. ej. price_out USD/Mtok)
    profile: np.ndarray  # (k, p) perfil de cada config (log precio, log tamaño, thinking, esfuerzo, ...)
    batch: np.ndarray  # (n,) id de lote de consultas
    events: list = field(default_factory=list)


def _knn(Xtr, Ytr, Xte, k=25):
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min(k, len(Xtr)), metric="cosine").fit(Xtr)
    _, idx = nn.kneighbors(Xte)
    return np.nanmean(Ytr[idx], axis=1)  # (n_te, k_cfg)


def _rows(snap, ids):
    return np.where(np.isin(snap.qids, list(ids)))[0]


# ------------------------------------------------------------------ routers
class StaticRouter:
    name = "static"

    def __init__(self, k=25):
        self.k = k

    def fit(self, s0: Snap, train_ids):
        r = _rows(s0, train_ids)
        self.X, self.Q, self.C = s0.X[r], s0.Q[r], s0.C[r]
        self.ids = list(s0.config_ids)
        return self

    def predict(self, snap: Snap, train_ids=None, f=0.0, seed=0):
        Qh = np.full((len(snap.X), len(snap.config_ids)), np.nan)
        Ch = Qh.copy()
        known = [(j, self.ids.index(c)) for j, c in enumerate(snap.config_ids) if c in self.ids]
        if known:
            q = _knn(self.X, self.Q, snap.X, self.k)
            c = _knn(self.X, self.C, snap.X, self.k)
            for j, j0 in known:
                Qh[:, j], Ch[:, j] = q[:, j0], c[:, j0]  # precios y calidad OBSOLETOS
        return Qh, Ch


class RefitRouter:
    name = "refit"

    def __init__(self, k=25):
        self.k = k

    def fit(self, s0, train_ids):
        return self

    def predict(self, snap, train_ids, f=1.0, seed=0):
        rows = _rows(snap, train_ids)
        rng = np.random.default_rng(seed)
        if f < 1.0:
            rows = rng.choice(rows, size=max(1, int(round(f * len(rows)))), replace=False)
        if len(rows) < 2:
            return np.full((len(snap.X), len(snap.config_ids)), np.nan), np.full(
                (len(snap.X), len(snap.config_ids)), np.nan
            )
        return _knn(snap.X[rows], snap.Q[rows], snap.X, self.k), _knn(
            snap.X[rows], snap.C[rows], snap.X, self.k
        )


class ProfileOnlineRouter:
    name = "profile_online"

    def __init__(self, k=25, n_prior=3, tau=5.0, temp=0.5):
        self.k, self.n_prior, self.tau, self.temp = k, n_prior, tau, temp

    def fit(self, s0, train_ids):
        r = _rows(s0, train_ids)
        self.X, self.Q, self.C = s0.X[r], s0.Q[r], s0.C[r]
        self.ids, self.prof, self.price0 = list(s0.config_ids), s0.profile, s0.price
        return self

    def predict(self, snap, train_ids, f=0.0, seed=0):
        n, k = len(snap.X), len(snap.config_ids)
        Qk, Ck = _knn(self.X, self.Q, snap.X, self.k), _knn(self.X, self.C, snap.X, self.k)  # (n, k0)
        Qb, Cb = np.zeros((n, k)), np.zeros((n, k))
        for j, c in enumerate(snap.config_ids):
            if c in self.ids:
                j0 = self.ids.index(c)
                Qb[:, j] = Qk[:, j0]
                Cb[:, j] = Ck[:, j0] * (snap.price[j] / self.price0[j0])
            else:  # arranque en frío: vecinos en el espacio de perfiles
                d = np.linalg.norm(self.prof - snap.profile[j], axis=1)
                o = np.argsort(d)[: self.n_prior]
                w = np.exp(-d[o] / self.temp)
                w /= w.sum()
                Qb[:, j] = Qk[:, o] @ w
                Cb[:, j] = (Ck[:, o] * (snap.price[j] / self.price0[o])[None, :]) @ w
        if f > 0:  # corrección online con feedback en sombra de una fracción f del tráfico de calibración
            rng = np.random.default_rng(seed)
            rows = _rows(snap, train_ids)
            A = rng.choice(rows, size=max(1, int(round(f * len(rows)))), replace=False)
            for j in range(k):
                m = ~np.isnan(snap.Q[A, j])
                nA = int(m.sum())
                if nA == 0:
                    continue
                Qb[:, j] += (snap.Q[A, j][m] - Qb[A, j][m]).sum() / (nA + self.tau)
                base = max(Cb[A, j][m].mean(), 1e-12)
                Cb[:, j] *= (snap.C[A, j][m].mean() * nA + self.tau * base) / ((nA + self.tau) * base)
        return np.clip(Qb, 0, 1), Cb


# ------------------------------------------------------------------ evaluación
def _curve(Qh, Ch, Q, C, rows, lams):
    Qh, Ch, Q, C = Qh[rows], Ch[rows], Q[rows], C[rows]
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sc = np.nanmax(np.nanmean(Ch, axis=0))
    sc = sc if np.isfinite(sc) and sc > 0 else 1.0
    pts = []
    for lam in lams:
        s = (1 - lam) * Qh - lam * Ch / sc
        s = np.where(np.isnan(s), -np.inf, s)
        ch = np.argmax(s, axis=1)
        idx = np.arange(len(ch))
        pts.append((C[idx, ch].mean(), np.nan_to_num(Q[idx, ch], nan=0.0).mean()))
    return np.array(pts)


def evaluate_snapshot(router, snap: Snap, train_ids, test_ids, f=0.0, seed=0, lams=None, subset="all"):
    lams = np.linspace(0, 1, 41) if lams is None else lams
    Qh, Ch = router.predict(snap, train_ids, f, seed)
    rows = _rows(snap, test_ids)
    if subset == "new":
        rows = rows[snap.batch[rows] != "b0"]
    elif subset == "old":
        rows = rows[snap.batch[rows] == "b0"]
    if len(rows) == 0:
        return None
    Q, C = np.nan_to_num(snap.Q[rows], nan=0.0), snap.C[rows]
    singles = np.c_[C.mean(0), Q.mean(0)]
    c_lo, c_hi = singles[:, 0].min(), singles[:, 0].max()
    o = M.oracle_choice(Q, C)
    oracle = np.array([[C[np.arange(len(o)), o].mean(), Q[np.arange(len(o)), o].mean()]])
    cur = _curve(Qh, Ch, snap.Q, snap.C, rows, lams)
    a_r, a_s = M.aiq(cur, c_lo, c_hi), M.aiq(singles, c_lo, c_hi)
    a_ceil = M.aiq(np.vstack([singles, oracle]), c_lo, c_hi)
    return dict(aiq=a_r, aiq_singles=a_s, aiq_ceiling=a_ceil, norm_gain=(a_r - a_s) / max(a_ceil - a_s, 1e-9))


def prequential(
    snaps, train_ids, test_ids, fractions=(0.0, 0.005, 0.01, 0.02, 0.05, 0.2, 1.0), seed=0, subset="all"
):
    """Devuelve DataFrame: snapshot, evento(s), router, f, aiq, norm_gain. Entrena en snaps[0]."""
    import pandas as pd

    s0 = snaps[0]
    static = StaticRouter().fit(s0, train_ids)
    prof = ProfileOnlineRouter().fit(s0, train_ids)
    refit = RefitRouter()
    rows = []
    for s in snaps[1:]:
        ev = ",".join(e["type"] for e in s.events) or "-"
        r = evaluate_snapshot(static, s, train_ids, test_ids, 0.0, seed, subset=subset)
        if r:
            rows.append(dict(snapshot=s.sid, events=ev, router="static", f=0.0, **r))
        for f in fractions:
            for rt in (refit, prof):
                if rt is refit and f == 0.0:
                    continue
                r = evaluate_snapshot(rt, s, train_ids, test_ids, f, seed, subset=subset)
                if r:
                    rows.append(dict(snapshot=s.sid, events=ev, router=rt.name, f=f, **r))
    return pd.DataFrame(rows)


def h4_summary(df, target=0.8):
    """H4: pérdida relativa de AIQ del estático frente al reentrenado completo, y fracción mínima de
    tráfico con la que cada router recupera `target` de esa pérdida (en norm_gain)."""
    out = []
    for sid, g in df.groupby("snapshot"):
        st = g[g.router == "static"].iloc[0]
        full = g[(g.router == "refit") & (g.f == 1.0)]
        if full.empty:
            continue
        full = full.iloc[0]
        gap = full.norm_gain - st.norm_gain
        rec = {}
        for rt in ("profile_online", "refit"):
            h = g[g.router == rt].sort_values("f")
            need = None
            for _, r in h.iterrows():
                if gap > 1e-9 and (r.norm_gain - st.norm_gain) / gap >= target:
                    need = float(r.f)
                    break
            rec[f"f_needed_{rt}"] = need
        out.append(
            dict(
                snapshot=sid,
                events=st.events,
                rel_loss_static=float(1 - st.aiq / full.aiq),
                ng_static=float(st.norm_gain),
                ng_full=float(full.norm_gain),
                gap_norm_gain=float(gap),
                **rec,
            )
        )
    import pandas as pd

    return pd.DataFrame(out)


# ------------------------------------------------------------------ featurización y simulador
def featurize_tfidf(texts, dim=64, seed=0):
    """Rasgos baratos (TF-IDF + SVD) para pruebas. En producción usa un modelo de embeddings fijo y
    NO lo cambies entre snapshots (cambiaría la representación y confundiría la deriva)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    Xt = TfidfVectorizer(max_features=5000).fit_transform(texts)
    d = min(dim, Xt.shape[1] - 1)
    return TruncatedSVD(n_components=d, random_state=seed).fit_transform(Xt)


def profile_vector(model_size_b, thinking, effort, budget, price_out):
    eff = {"nothink": 0, "think_low": 1, "think_mid": 2, "think_high": 3}.get(effort, 0)
    return np.array(
        [
            np.log10(max(price_out, 1e-6)),
            np.log10(max(model_size_b, 0.1)),
            float(thinking),
            eff / 3.0,
            np.log10(budget) / 4 if budget and budget == budget else 1.0,
        ]
    )


def simulate_drift(n=3000, seed=0, n_new=500):
    """Secuencia de snapshots sintética con eventos: s1 regresión de revisión, s2 modelo nuevo,
    s3 bajada de precio, s4 baja de un modelo, s5 lote de consultas nuevas (desplazadas)."""
    rng = np.random.default_rng(seed)
    base = dict(
        a=dict(size=1, think=0, eff="nothink", bud=500, skill=0.30, price=0.05),
        b=dict(size=8, think=0, eff="nothink", bud=500, skill=0.50, price=0.2),
        c=dict(size=8, think=1, eff="think_low", bud=None, skill=0.62, price=0.5),
        d=dict(size=30, think=0, eff="nothink", bud=500, skill=0.70, price=1.0),
        e=dict(size=70, think=1, eff="think_mid", bud=None, skill=0.82, price=3.0),
    )
    d0 = np.clip(rng.beta(2, 2, n), 0, 1)
    dom = rng.integers(0, 4, n)

    def feats(d, dom):
        return np.c_[
            np.ones(len(d)),
            3 * (d + rng.normal(0, 0.05, len(d))),
            np.eye(4)[dom] * 0.5,
            rng.normal(0, 0.05, (len(d), 4)),
        ]

    X0 = feats(d0, dom)
    qids0 = np.array([f"q{i}" for i in range(n)])
    dn = np.clip(rng.beta(3, 2, n_new) + 0.1, 0, 1)
    domn = rng.integers(0, 4, n_new)
    Xn = feats(dn, domn)
    qidsn = np.array([f"n{i}" for i in range(n_new)])
    tok = {k: rng.integers(150, 600, n + n_new) for k in list(base) + ["f"]}
    cfg_now, snaps, _events_all = {k: dict(v) for k, v in base.items()}, [], []
    plan = {
        1: [dict(type="new_revision", id="c", dskill=-0.18)],
        2: [
            dict(
                type="add_model",
                id="f",
                spec=dict(size=14, think=1, eff="think_mid", bud=None, skill=0.85, price=0.9),
            )
        ],
        3: [dict(type="price_change", id="e", factor=0.4)],
        4: [dict(type="retire_model", id="a")],
        5: [dict(type="new_query_batch")],
    }
    batch_active = False
    for si in range(6):
        ev = plan.get(si, [])
        for e in ev:
            if e["type"] == "new_revision":
                cfg_now[e["id"]]["skill"] += e["dskill"]
            elif e["type"] == "add_model":
                cfg_now[e["id"]] = dict(e["spec"])
            elif e["type"] == "price_change":
                cfg_now[e["id"]]["price"] *= e["factor"]
            elif e["type"] == "retire_model":
                cfg_now.pop(e["id"])
            elif e["type"] == "new_query_batch":
                batch_active = True
        d = np.r_[d0, dn] if batch_active else d0
        X = np.r_[X0, Xn] if batch_active else X0
        qids = np.r_[qids0, qidsn] if batch_active else qids0
        batch = (
            np.r_[np.array(["b0"] * n), np.array(["b1"] * n_new)] if batch_active else np.array(["b0"] * n)
        )
        ids = list(cfg_now)
        N = len(d)
        Q = np.zeros((N, len(ids)))
        C = np.zeros((N, len(ids)))
        for j, k in enumerate(ids):
            sp = cfg_now[k]
            p = 1 / (1 + np.exp(-16 * (sp["skill"] - d)))
            Q[:, j] = (np.random.default_rng(seed + 100 * si + j).random(N) < p).astype(float)
            C[:, j] = tok[k][:N] * sp["price"] / 1e3
        prof = np.stack(
            [
                profile_vector(
                    cfg_now[k]["size"],
                    cfg_now[k]["think"],
                    cfg_now[k]["eff"],
                    cfg_now[k]["bud"],
                    cfg_now[k]["price"],
                )
                for k in ids
            ]
        )
        snaps.append(
            Snap(
                f"s{si:02d}",
                ids,
                X,
                qids,
                Q,
                C,
                np.array([cfg_now[k]["price"] for k in ids]),
                prof,
                batch,
                ev,
            )
        )
    return snaps, qids0
