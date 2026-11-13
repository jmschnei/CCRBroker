import os, sys, json, tempfile, io, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd, yaml
from routedata import drift as D, splits as SP, pricing as PR
from routedata.schema import save_table
from generate import driftroute as DR, run_phase3 as P3

def test_drift_simulation_qualitative():
    snaps, q0 = D.simulate_drift(seed=0); tr, te = DR.split_train_test(pd.DataFrame({"query_id": q0}), 0.5, 0)
    tests = set(te) | {q for s in snaps for q in s.qids[s.batch != "b0"]}
    assert not set(tr) & set(te)
    df = D.prequential(snaps, set(tr), tests); hs = D.h4_summary(df)
    assert (hs.rel_loss_static > 0).all()                                     # el estático siempre pierde frente a reentrenar
    assert (hs.ng_full > hs.ng_static).all()
    p0 = df[(df.router == "profile_online") & (df.f == 0.0)].set_index("snapshot").norm_gain
    p1 = df[(df.router == "profile_online") & (df.f == 0.01)].set_index("snapshot").norm_gain
    assert (p1 >= p0 - 1e-9).all() and (p1 > p0).mean() >= 0.8               # 1 % de calibración ayuda casi siempre
    new = D.prequential(snaps, set(tr), tests, fractions=(0.0, 1.0), subset="new"); assert len(new) and (new.snapshot == "s05").all()

def test_events_comparability_pricing():
    pool = {"models": {"a": dict(revision="r1", thinking=False, size_b=8, tp=1)}, "gpu": {}}
    p, ev = DR.apply_pool_events(pool, "s01", add=["b=r9,14,1,2"], price_factor=["a=0.5"], revision=["a=r2"])
    assert p["models"]["b"]["tp"] == 2 and p["models"]["a"]["price_factor"] == 0.5 and p["models"]["a"]["revision"] == "r2"
    assert {e["type"] for e in ev} == {"add_model", "price_change", "new_revision"} and pool["models"]["a"]["revision"] == "r1"
    m0 = DR.SnapshotManifest("s00", "d", {}, prompt_hash="x", gen_params={"t": 0.2}, queries_hash="h")
    m1 = DR.SnapshotManifest("s01", "d", {}, prompt_hash="y", gen_params={"t": 0.2}, queries_hash="h2")
    w = DR.comparability_check([m0, m1]); assert any("prompt_hash" in x for x in w) and any("consultas" in x for x in w)
    cf = pd.DataFrame({"config_id": ["a|n|1", "b|n|1"], "model": ["a", "b"], "price_in": 0.0, "price_out": 0.0, "price_source": "", "price_date": ""})
    resp = pd.DataFrame({"query_id": ["q"] * 2, "config_id": ["a|n|1", "b|n|1"], "out_tokens": [800, 100], "latency_s": [1.0, 1.0]})
    tp = PR.throughput_from_responses(resp, cf); assert tp["a"]["decode_tps"] == 800
    c2 = PR.reprice_configs(cf, tp, 1.8, {"b": 4}); assert (c2.price_out > 0).all() and c2.set_index("model").price_out["b"] > 0

def _queries(n=240, seed=0):
    rng = np.random.default_rng(seed); d = rng.random(n); dom = rng.integers(0, 3, n)
    return pd.DataFrame({"query_id": [f"q{i}" for i in range(n)], "text": [f"question level{int(x*10)} level{int(x*10)} topic{t} item{i}" for i, (x, t) in enumerate(zip(d, dom))],
        "options": [json.dumps(list("abcd"))] * n, "answer": ["A"] * n, "domain": [f"d{t}" for t in dom], "difficulty": ["m"] * n,
        "source_dataset": "s", "language": "en", "cluster_id": np.arange(n),
        "meta": [json.dumps({"task_type": "mc", "difficulty": float(x), "wrong": "Z"}) for x in d]})

def test_cli_end_to_end():
    with tempfile.TemporaryDirectory() as t:
        save_table(_queries(), os.path.join(t, "q")); dd = os.path.join(t, "drift")
        pool = yaml.safe_load(open("configs/pool_openweights.yaml")); pool["models"] = dict(list(pool["models"].items())[:3])
        pool["judge_freeze"] = {"models": {}, "prompt_hash": "j"}; yaml.safe_dump(pool, open(os.path.join(t, "p0.yaml"), "w"))
        cfg = yaml.safe_load(open("configs/driftroute.yaml")); cfg["subset_queries"] = 200; cfg["fractions"] = [0.0, 0.05, 1.0]
        cfg["grid"] = {"efforts": ["nothink", "think_low"], "budgets_tokens": [500, None]}; cp = os.path.join(t, "cfg.yaml"); yaml.safe_dump(cfg, open(cp, "w"))
        c = ["--cfg", cp, "--drift-dir", dd]; q = io.StringIO()
        with contextlib.redirect_stdout(q):
            P3.main(["init", *c, "--queries", os.path.join(t, "q")])
            P3.main(["snapshot", *c, "--snapshot", "s00", "--pool", os.path.join(t, "p0.yaml"), "--backend", "mock", "--date", "2026-11-01"])
            P3.main(["new-pool", *c, "--base-pool", os.path.join(t, "p0.yaml"), "--out", os.path.join(t, "p1.yaml"), "--snapshot", "s01", "--add", "extra/Model-14B=rev1,14,1,1"])
            P3.main(["snapshot", *c, "--snapshot", "s01", "--pool", os.path.join(t, "p1.yaml"), "--backend", "mock", "--date", "2026-12-01"])
            first = list(pool["models"])[0]
            P3.main(["new-pool", *c, "--base-pool", os.path.join(t, "p1.yaml"), "--out", os.path.join(t, "p2.yaml"), "--snapshot", "s02", "--price-factor", f"{first}=0.5", "--revision", f"{list(pool['models'])[1]}=revNEW"])
            P3.main(["snapshot", *c, "--snapshot", "s02", "--pool", os.path.join(t, "p2.yaml"), "--backend", "mock", "--date", "2027-01-01"])
            P3.main(["check", *c]); P3.main(["report", *c]); P3.main(["eval", *c])
        out = q.getvalue()
        assert "snapshot s02 listo" in out and "comparables" in out
        for f in ("DRIFT_REPORT.md", "e4_results.csv", "E4_REPORT.md", "subset/split.json", "s01/manifest_s01.json"): assert os.path.exists(os.path.join(dd, f)), f
        res = pd.read_csv(os.path.join(dd, "e4_results.csv")); assert {"static", "refit", "profile_online"} <= set(res.router) and res.norm_gain.notna().all()
        man = json.load(open(os.path.join(dd, "s02/manifest_s02.json"))); assert {e["type"] for e in man["events"]} == {"price_change", "new_revision"}
        # el precio del modelo con factor 0.5 debe reflejarse en configs de s02 frente a s01
        from routedata.schema import load_table
        p1, p2 = load_table(os.path.join(dd, "s01/configs")), load_table(os.path.join(dd, "s02/configs"))
        a1 = p1[p1.model == first].price_out.iloc[0]; a2 = p2[p2.model == first].price_out.iloc[0]; assert a2 < a1 * 0.9
        # los snapshots comparten consultas del subconjunto
        assert DR.ids_hash(load_table(os.path.join(dd, "s00/queries")).query_id) == DR.ids_hash(load_table(os.path.join(dd, "s02/queries")).query_id)

if __name__ == "__main__":
    for f in (test_drift_simulation_qualitative, test_events_comparability_pricing, test_cli_end_to_end):
        f(); print("OK ", f.__name__)
