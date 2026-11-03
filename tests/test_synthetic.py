"""Pruebas de extremo a extremo con datos sintéticos (sin GPU, sin red).
Ejecuta:  python tests/test_synthetic.py   (o pytest, si está instalado)."""
import os, sys, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from routedata import metrics as M, stats as S, splits as SP, conformal as CF
from routedata.schema import validate, to_matrices, save_table, load_table
from routedata.baselines import trivial_baselines, KNNRouter
from routedata.pricing import derive_selfhost_prices, scale_prices, request_cost
from routedata.adapters import assert_trainable, EvalOnlyError
from generate.backends import MockBackend, GenRequest
from generate import configroute as CR, efforts as EF, advroute as AD, agentroute as AG, driftroute as DR, verifiers as V

def synth_matrices(n=3000, k=8, seed=0):
    rng = np.random.default_rng(seed)
    dom = rng.integers(0, 5, n); d = np.clip(rng.beta(2, 2, n) + 0.1 * (dom - 2) / 2, 0, 1)
    skill = np.linspace(0.25, 0.95, k) + rng.normal(0, 0.03, k)
    skill[2] += 0.15 * (dom == 1).mean()  # un poco de especialización
    P = 1 / (1 + np.exp(-6 * (skill[None, :] - d[:, None] + 0.1 * (dom[:, None] == np.arange(k)[None, :] % 5))))
    Q = (rng.random((n, k)) < P).astype(float)
    price = np.geomspace(0.05, 5.0, k); tok = rng.integers(150, 600, (n, k))
    C = tok * price[None, :] / 1e6 * 1e3
    X = np.c_[d + rng.normal(0, 0.08, n), np.eye(5)[dom] * 0.6, rng.normal(0, 1, (n, 3)) * 0.05]
    return Q, C, X, dom, d, P

def test_metrics_and_baselines():
    Q, C, X, dom, d, P = synth_matrices()
    tr, te = SP.iid_split(len(Q), 0.3, seed=1)
    b = trivial_baselines(Q[te], C[te], groups=dom[te])
    q = {k: M.mean_quality_cost(Q[te], C[te], v) for k, v in b.items()}
    assert q["oracle"][0] >= q["best_single"][0] >= q["random"][0] - 1e-9
    assert q["oracle"][1] <= q["best_single"][1] + 1e-9           # el oráculo es más barato que el mejor único
    r = KNNRouter(40).fit(X[tr], Q[tr], C[tr]); Qh, Ch = r.predict(X[te])
    cur = M.deferral_curve(Qh, Ch, Q[te], C[te])
    bs = b["best_single"][0]; bq, bc = q["best_single"]
    a = M.audc(cur, cost_norm=float(C[te].mean(axis=0).max())); assert 0 < a <= 1
    assert M.qnc(cur, bq, bc) <= 1.0
    perf, save = M.perf_gain_cost_save(cur[:, ::-1] if False else np.c_[cur[:, 0], cur[:, 1]], (bc, bq))
    assert M.gap_at_oracle(Q[te], C[te], b["best_single"], dom[te]) > 0
    pts = np.c_[C[te].mean(0), Q[te].mean(0)]; fr = M.pareto_frontier(pts)
    assert M.hypervolume_2d(pts, ref_cost=pts[:, 0].max() * 1.01) > 0 and M.aiq(pts) > 0
    return dict(best_single=q["best_single"], oracle=q["oracle"], audc=a, perf_gain=perf, cost_save=save)

def test_stats_splits():
    rng = np.random.default_rng(0); v = rng.random(500) < 0.7
    m, lo, hi = S.bootstrap_ci(v.astype(float), n_boot=2000); assert lo < m < hi
    p, n01, n10 = S.mcnemar_exact(v, v); assert p == 1.0
    adj, rej = S.holm([0.001, 0.04, 0.2]); assert rej[0] and not rej[2]
    g = np.repeat(np.arange(100), 10); tr, te = SP.iid_split(1000, 0.3, 0, groups=g)
    assert not set(g[tr]) & set(g[te])                              # sin fuga entre clusters
    assert len(SP.leave_model_out(8, 2, 3)[0][1]) == 2
    assert len(list(SP.prequential(["s0", "s1", "s2"]))) == 2

def test_conformal_risk():
    risks = []
    for seed in range(40):
        Q, C, X, dom, d, P = synth_matrices(n=2000, k=6, seed=seed)
        rng = np.random.default_rng(seed)
        Phat = np.clip(P + rng.normal(0, 0.08, P.shape), 0, 1)       # predictor imperfecto
        order = np.argsort(C.mean(0)); cal, te = SP.iid_split(len(Q), 0.5, seed)
        t = CF.crc_threshold(Q[cal], Phat[cal], order, alpha=0.15)
        if t is None: continue
        risks.append(CF.risk_at(Q[te], Phat[te], order, t))
    assert len(risks) > 30 and np.mean(risks) <= 0.15 + 0.01, (len(risks), np.mean(risks))
    return dict(mean_test_risk=float(np.mean(risks)), alpha=0.15, n_trials=len(risks))

def _tiny_queries(n=40):
    rng = np.random.default_rng(1)
    return pd.DataFrame({"query_id": [f"q{i}" for i in range(n)], "text": [f"question {i}" for i in range(n)],
        "options": [json.dumps(["x", "y", "z", "w"])] * n, "answer": rng.choice(list("ABCD"), n),
        "domain": rng.choice(["math", "science"], n), "source_dataset": "synthetic", "difficulty": "m",
        "meta": [json.dumps({"task_type": "mc", "difficulty": float(rng.random()), "wrong": "Z"})] * n})

def test_configroute_pipeline_and_resume():
    q = _tiny_queries(); 
    models = {"mockA": dict(revision="r0", price_in=0.1, price_out=0.2, thinking=True),
              "mockB": dict(revision="r0", price_in=0.5, price_out=1.0, thinking=False)}
    cfgs = CR.build_config_grid(models, budgets=(50, None), efforts=("nothink", "think_low"))
    skills = {"mockA": 0.4, "mockB": 0.8}
    fac = lambda m, rev: MockBackend(m, skill=skills[m])
    with tempfile.TemporaryDirectory() as d:
        r1 = CR.run(q, cfgs, fac, d, batch=16, log=lambda *_: None); n1 = len(r1)
        r2 = CR.run(q, cfgs, fac, d, batch=16, log=lambda *_: None)
        assert n1 == len(r2) == len(q) * len(cfgs)                   # reanudación: no duplica
        assert validate(r1, "responses") == [] and validate(cfgs, "configs") == []
        Q, C, T = to_matrices(q, cfgs, r1); assert Q.shape == (40, len(cfgs)) and not np.isnan(Q).any()
        # esfuerzo thinking debe consumir más tokens que nothink (A)
        ra = r1.merge(cfgs[["config_id", "effort", "model"]], on="config_id"); ra = ra[ra.model == "mockA"]
        assert ra[ra.effort == "think_low"].out_tokens.mean() > 0
        # el modelo fuerte acierta más que el débil
        assert r1[r1.config_id.str.startswith("mockB")].correct.mean() > r1[r1.config_id.str.startswith("mockA|nothink")].correct.mean()

def test_pricing_verifiers_eval_only():
    p = derive_selfhost_prices(1.8, 1, decode_tok_per_s=2500, prefill_tok_per_s=20000)
    assert p.usd_per_mtok_out > p.usd_per_mtok_in > 0 and scale_prices(p, 0.5).usd_per_mtok_out < p.usd_per_mtok_out
    assert request_cost(1000, 500, p) > 0
    assert V.verify_mc("Reasoning...\nFinal answer: C", "c") == 1.0
    assert V.verify_numeric("so \\boxed{42}\nFinal answer: 42.0", 42) == 1.0
    assert V.run_code_tests("def f(x): return x+1", "assert f(1)==2") == 1.0
    try:
        assert_trainable(pd.DataFrame({"eval_only": [True]})); raise AssertionError("debió fallar")
    except EvalOnlyError:
        pass

def test_adv_agent_drift():
    opts, letter = AD.shuffle_options(["a", "b", "c", "d"], "B", seed=3)
    assert opts["ABCD".index(letter)] == "b"
    assert len(AD.typos("hello world foo bar", 0.5, 1)) == len("hello world foo bar")
    vocab = ["alpha", "beta", "URGENT", "prove", "rigorous", "zzz"]
    score = lambda texts: np.array([0.2 + 0.15 * sum(w in ("URGENT", "rigorous") for w in t.split()) for t in texts]).clip(0, 1)
    g, s0, s1 = AD.gadget_search(score, ["what is 2+2", "capital of France"], vocab, 6, 120, 0); assert s1 > s0
    be = {"small": MockBackend("small", 0.4), "big": MockBackend("big", 0.9)}
    ep = AG.run_episode(AG.ToyEnv(horizon=6, difficulty=0.5), be, AG.switch_every_k(["small", "big"], 2), "e0", "sw2")
    tot = ep.totals({"small": (0.1, 0.2), "big": (1.0, 2.0)}); assert tot["switches"] >= 1
    assert any(s["switch_prefill_tokens"] > 0 for s in ep.steps if s["switched"])
    fixed = AG.run_episode(AG.ToyEnv(6), be, AG.fixed_policy("big"), "e1", "big")
    assert fixed.totals({"small": (0.1, 0.2), "big": (1.0, 2.0)})["switches"] == 0
    ra = pd.DataFrame({"query_id": [f"q{i}" for i in range(50)], "config_id": "c", "correct": [1.0] * 50, "out_tokens": 100})
    rb = ra.assign(correct=[1.0] * 40 + [0.0] * 10)
    rep = DR.drift_report(ra, rb); assert rep.delta_acc.iloc[0] < 0 and rep.p_mcnemar.iloc[0] < 0.05
    cf = pd.DataFrame({"config_id": ["a|x|1", "b|x|1"], "model": ["a", "b"], "price_in": 1.0, "price_out": 2.0})
    out = DR.apply_events(cf, [{"type": "price_change", "model": "a", "factor": 0.5}, {"type": "retire_model", "model": "b"}])
    assert len(out) == 1 and out.price_out.iloc[0] == 1.0

def test_effort_budget_forcing():
    be = MockBackend("m", 0.6); reqs = [GenRequest([{"role": "user", "content": "q"}], meta={"query_id": f"q{i}", "answer": "A", "difficulty": 0.9}) for i in range(8)]
    lo = EF.run_with_effort(be, reqs, EF.EFFORTS["think_low"]); hi = EF.run_with_effort(be, reqs, EF.EFFORTS["think_high"])
    assert np.mean([g.out_tokens for g in hi]) >= np.mean([g.out_tokens for g in lo])

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        r = t(); print(f"OK  {t.__name__}", "" if r is None else r)
    print(f"\n{len(tests)} pruebas OK")
