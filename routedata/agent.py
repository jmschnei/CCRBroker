"""Routers y evaluación para agentes (Fase 4 · RQ5 · H5).

TaskRouter  : elige UN modelo por tarea a partir de la descripción inicial (baseline 'por tarea').
StepRouter  : elige el modelo en CADA paso con utilidades contrafactuales aprendidas del estado actual
              (observación + paso) y un coste de paso EXACTO calculado con el modelo de caché; el coste
              de cambio de modelo aparece como prefill extra no cacheado (penalty=True) o se ignora (False).
Evaluación  : éxito/coste/latencia por episodio recalculables bajo distintos modos de caché y precios;
              comparaciones pareadas por tarea (bootstrap, McNemar, Holm).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from . import metrics as M
from .stats import mcnemar_exact, holm, bootstrap_ci
from generate.agentroute import candidate_prefill, approx_context, steps_cost, Episode

# ------------------------------------------------------------------ rasgos (sin ajuste: reproducibles)
def featurize_text(texts, n_features: int = 512):
    from sklearn.feature_extraction.text import HashingVectorizer
    return HashingVectorizer(n_features=n_features, alternate_sign=False, norm="l2", ngram_range=(1, 2)).transform(list(texts)).toarray()

def featurize_state(obs: str, t: int, max_steps: int = 30):
    return np.r_[featurize_text([obs])[0], t / max_steps]

# ------------------------------------------------------------------ routers
class TaskRouter:
    def __init__(self, k: int = 15): self.k = k
    def fit(self, X, S, C):
        from sklearn.neighbors import NearestNeighbors
        self.nn = NearestNeighbors(n_neighbors=min(self.k, len(X)), metric="cosine").fit(X); self.S, self.C = np.asarray(S), np.asarray(C); return self
    def predict(self, x):
        _, i = self.nn.kneighbors(np.asarray(x).reshape(1, -1)); return self.S[i[0]].mean(0), self.C[i[0]].mean(0)

class StepRouter:
    """Ridge por modelo sobre rasgos del estado: utilidad esperada y tokens de salida esperados."""
    def __init__(self, models, alpha: float = 1.0): self.models, self.alpha = list(models), alpha
    def fit(self, rows: pd.DataFrame, max_steps: int = 30):
        from sklearn.linear_model import Ridge
        self.reg_u, self.reg_o, self.reg_c = {}, {}, {}
        for m in self.models:
            g = rows[rows.model == m]
            X = np.stack([featurize_state(o, t, max_steps) for o, t in zip(g.obs, g.t)])
            self.reg_u[m] = Ridge(alpha=self.alpha).fit(X, g.utility.values)
            self.reg_o[m] = Ridge(alpha=self.alpha).fit(X, g.out_tokens_first.values)
            self.reg_c[m] = Ridge(alpha=self.alpha).fit(X, g.cost_after.values)
        self.max_steps = max_steps; return self
    def predict(self, obs, t):
        x = featurize_state(obs, t, self.max_steps).reshape(1, -1)
        return {m: (float(np.clip(self.reg_u[m].predict(x)[0], 0, 1)), float(max(self.reg_o[m].predict(x)[0], 1.0)), float(max(self.reg_c[m].predict(x)[0], 0.0))) for m in self.models}

# ------------------------------------------------------------------ políticas
def task_router_policy(router: TaskRouter, models, lam: float, featurize=featurize_text):
    """Elige un modelo al inicio: argmax Ŝ − λ·Ĉ_USD y lo mantiene."""
    st = {}
    def pol(t, h, o):
        if t == 0:
            S, C = router.predict(featurize([o])[0]); st["m"] = models[int(np.argmax(S - lam * C))]
        return st["m"]
    return pol

def step_router_policy(router: StepRouter, price: dict, lam: float, cache_mode: str = "prefix", retention: str = "keep",
                       penalty: bool = True, stick_bonus_usd: float = 0.0):
    """Score(m) = Û(m) − λ·[coste_paso_exacto(m) + Ĉ_posterior(m)] (+ bonus de permanencia). Con penalty=True, coste_paso usa el prefill REAL no
    cacheado de m (cambiar de modelo lo encarece); con penalty=False se ignora el estado de caché (ingenuo)."""
    models = router.models
    def pol(t, h, o):
        pred = router.predict(o, t); ctx = approx_context(h, o); prev = h[-1]["model"] if h else None; best, bs = None, -1e18
        for m in models:
            u, out, after = pred[m]; p = price[m]
            pre = candidate_prefill(h, m, ctx, cache_mode, retention) if penalty else min(ctx, len(o) // 4 + 20)
            usd = (pre * p["in"] + out * p["out"]) / 1e6 + after - (stick_bonus_usd if m == prev else 0.0)
            s = u - lam * usd
            if s > bs: best, bs = m, s
        return best
    return pol

DEFAULT_LAMS = (0.0, 10.0, 100.0, 1e3, 1e4, 1e5)

# ------------------------------------------------------------------ evaluación
def episodes_to_frame(episodes, price: dict, cache_modes=("none", "prefix"), retention: str = "keep"):
    rows = []
    for e in episodes:
        ep = Episode(e["episode_id"], e["policy"], e["steps"], e["success"], e.get("task_id", ""), e.get("env_name", ""))
        r = dict(task_id=ep.task_id, policy=ep.policy, success=ep.success, steps=len(ep.steps), switches=sum(s["switched"] for s in ep.steps))
        for cm in cache_modes:
            c, l = steps_cost(ep.steps, price, cm, retention); r[f"cost_{cm}"], r[f"lat_{cm}"] = c, l
        rows.append(r)
    return pd.DataFrame(rows)

def summarize(df: pd.DataFrame, cache_modes=("none", "prefix"), n_boot: int = 2000, seed: int = 0):
    out = []
    for pol, g in df.groupby("policy"):
        r = dict(policy=pol, n_tasks=len(g), success=g.success.mean(), steps=g.steps.mean(), switches=g.switches.mean())
        r["success_lo"], r["success_hi"] = bootstrap_ci(g.success.values, n_boot=n_boot, seed=seed)[1:]
        for cm in cache_modes:
            r[f"cost_{cm}"] = g[f"cost_{cm}"].mean(); r[f"lat_{cm}"] = g[f"lat_{cm}"].mean()
            r[f"cost_per_success_{cm}"] = g[f"cost_{cm}"].sum() / max(g.success.sum(), 1e-9)
        out.append(r)
    return pd.DataFrame(out)

def compare_policies(df: pd.DataFrame, base: str, others, cache_mode: str = "prefix", n_boot: int = 5000, seed: int = 0):
    """Diferencias pareadas por tarea frente a `base`: Δéxito y Δcoste (IC bootstrap), McNemar sobre éxito y Holm."""
    rng = np.random.default_rng(seed); b = df[df.policy == base].set_index("task_id"); rows = []
    for o in others:
        g = df[df.policy == o].set_index("task_id"); ids = b.index.intersection(g.index)
        ds, dc = (g.loc[ids, "success"] - b.loc[ids, "success"]).values, (g.loc[ids, f"cost_{cache_mode}"] - b.loc[ids, f"cost_{cache_mode}"]).values
        idx = rng.integers(0, len(ids), size=(n_boot, len(ids)))
        p, n01, n10 = mcnemar_exact(b.loc[ids, "success"].values, g.loc[ids, "success"].values)
        rows.append(dict(policy=o, vs=base, n=len(ids), d_success=ds.mean(), d_success_lo=np.quantile(ds[idx].mean(1), .025), d_success_hi=np.quantile(ds[idx].mean(1), .975),
                         d_cost=dc.mean(), d_cost_lo=np.quantile(dc[idx].mean(1), .025), d_cost_hi=np.quantile(dc[idx].mean(1), .975), p_mcnemar=p))
    res = pd.DataFrame(rows)
    if len(res): res["p_holm"], res["signif"] = holm(res.p_mcnemar.values)
    return res

def headroom(df_fixed: pd.DataFrame, cache_mode: str = "prefix"):
    """¿Hay margen para enrutar? Éxito del mejor modelo fijo vs oráculo POR TAREA (el más barato que triunfa).
    Si el oráculo por tarea no supera al mejor fijo, el routing por tarea no puede ayudar (el por paso puede)."""
    d = df_fixed.copy(); d["model"] = d.policy.str.replace("fixed:", "", regex=False); c = f"cost_{cache_mode}"
    per = d.groupby("model").agg(success=("success", "mean"), cost=(c, "mean"))
    best = per.success.idxmax(); piv_s, piv_c = d.pivot(index="task_id", columns="model", values="success"), d.pivot(index="task_id", columns="model", values=c)
    ok = piv_s >= 0.5; cost_ok = piv_c.where(ok, np.inf); o_cost = cost_ok.min(axis=1); solved = np.isfinite(o_cost)
    return dict(per_model=per.round(4), best_fixed=best, oracle_task_success=float(solved.mean()),
                oracle_task_cost=float(o_cost[solved].mean()) if solved.any() else float("nan"),
                gap_success=float(solved.mean() - per.success.max()), cost_ratio_oracle_vs_best=float(o_cost[solved].mean() / per.loc[best, "cost"]) if solved.any() else float("nan"))
