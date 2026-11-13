"""CLI de la Fase 2.  Uso (desde la raíz del kit):

  python -m generate.run_phase2 build-queries --cfg configs/configroute_pilot.yaml --out data/queries [--open-ended abiertas.jsonl]
  python -m generate.run_phase2 pilot   --pool configs/pool_openweights.yaml --cfg configs/configroute_pilot.yaml --queries data/queries --out data/pilot [--backend vllm|mock] [--models m1,m2]
  python -m generate.run_phase2 judge   --run-dir data/pilot --pool configs/pool_openweights.yaml [--backend vllm|mock]
  python -m generate.run_phase2 audit-export --run-dir data/pilot --n 150
  python -m generate.run_phase2 audit-import --run-dir data/pilot --sheet data/pilot/audit/audit_sheet_filled.csv
  python -m generate.run_phase2 report  --run-dir data/pilot --pool configs/pool_openweights.yaml --cfg configs/configroute_pilot.yaml
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, pandas as pd, yaml
from routedata.schema import save_table, load_table
from . import configroute as CR, phase2 as P2, build_queries as BQ
from .backends import MockBackend, GenResult

class _JudgeMock:
    name = "judge-mock"
    def generate(self, reqs): return [GenResult('{"score": 0.7}', 10, 5) for _ in reqs]

def _pool(path): return yaml.safe_load(open(path))
def _factory(kind, pool):
    if kind == "mock":
        sk = {m: 0.3 + 0.6 * i / max(1, len(pool["models"]) - 1) for i, m in enumerate(pool["models"])}
        return lambda m, rev: MockBackend(m, skill=sk[m])
    from .backends import VLLMBackend
    return lambda m, rev: VLLMBackend(m, rev, tensor_parallel_size=pool["models"][m].get("tp", 1))

def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build-queries", "pilot", "judge", "audit-export", "audit-import", "report"):
        p = sub.add_parser(name)
        for a in ("--cfg", "--pool", "--queries", "--out", "--run-dir", "--sheet", "--open-ended", "--models", "--backend"):
            p.add_argument(a)
        p.add_argument("--n", type=int); p.add_argument("--n-full", type=int)
    a = ap.parse_args(argv); cfg = yaml.safe_load(open(a.cfg)) if a.cfg else {}

    if a.cmd == "build-queries":
        src = {n: BQ.load_source(n, cfg.get("seed", 0)) for n in cfg["sources_quota"] if n in BQ.SOURCES}
        if a.open_ended:
            src["open_ended"] = pd.read_json(a.open_ended, lines=True)   # debe cumplir el esquema `queries`
        q = BQ.drop_duplicates_across(CR.sample_queries(src, cfg["sources_quota"], seed=cfg.get("seed", 0)))
        print(save_table(q, os.path.join(a.out, "queries")), len(q))

    elif a.cmd == "pilot":
        pool = _pool(a.pool); q = load_table(a.queries + "/queries" if os.path.isdir(a.queries) else a.queries)
        q = q.sample(n=min(cfg["n_pilot"], len(q)), random_state=cfg.get("seed", 0))
        models = {m: dict(revision=s["revision"], price_in=0.0, price_out=0.0, thinking=s["thinking"]) for m, s in pool["models"].items()
                  if not a.models or m in a.models.split(",")}
        cfgs = CR.build_config_grid(models, budgets=tuple(cfg["budgets_tokens"]), efforts=tuple(cfg["efforts"]),
                                    hardware=pool["gpu"]["type"], price_date=pool["gpu"]["price_date"])
        CR.run(q, cfgs, _factory(a.backend or "vllm", pool), a.out, run_id="pilot", batch=64, seed=cfg.get("seed", 0))
        print("piloto completado en", a.out, "(precios en 0: derívalos con `report` tras medir tok/s)")

    elif a.cmd == "judge":
        pool = _pool(a.pool); q = load_table(os.path.join(a.run_dir, "queries")); r = CR.load_responses(a.run_dir)
        if (a.backend or "vllm") == "mock":
            judges = [_JudgeMock()]
        else:
            from .backends import VLLMBackend
            judges = [VLLMBackend(j, "main") for j in pool["judges"]]   # fija revisión real de cada juez
        lab = P2.apply_judges(q, r, judges); lab.to_json(os.path.join(a.run_dir, "labels_judge.jsonl"), orient="records", lines=True)
        print(len(lab), "respuestas juzgadas")

    elif a.cmd == "audit-export":
        q = load_table(os.path.join(a.run_dir, "queries")); r = CR.load_responses(a.run_dir)
        lab = pd.read_json(os.path.join(a.run_dir, "labels_judge.jsonl"), lines=True)
        P2.export_audit(P2.overlay_labels(r, lab), q, lab, n=a.n or 150, out_dir=os.path.join(a.run_dir, "audit"))
        print("hoja ciega:", os.path.join(a.run_dir, "audit/audit_sheet.csv"))

    elif a.cmd == "audit-import":
        met, merged = P2.import_audit(a.sheet, os.path.join(a.run_dir, "audit/audit_key_PRIVATE.csv"))
        P2.human_label_layer(merged).to_json(os.path.join(a.run_dir, "labels_human.jsonl"), orient="records", lines=True)
        json.dump(met, open(os.path.join(a.run_dir, "judge_vs_human.json"), "w"), indent=2); print(met)

    elif a.cmd == "report":
        pool = _pool(a.pool); q = load_table(os.path.join(a.run_dir, "queries")); c = load_table(os.path.join(a.run_dir, "configs"))
        r = CR.load_responses(a.run_dir)
        for f in ("labels_judge.jsonl", "labels_human.jsonl"):          # humano prevalece sobre juez
            p = os.path.join(a.run_dir, f)
            if os.path.exists(p): r = P2.overlay_labels(r, pd.read_json(p, lines=True))
        tp = {m: s.get("tp", 1) for m, s in pool["models"].items()}
        rep = P2.analyze_pilot(q, c, r, n_full=a.n_full or cfg.get("n_full_target", 8000), tp_by_model=tp, gpu_usd_hr=pool["gpu"]["usd_per_hour"])
        jm = json.load(open(os.path.join(a.run_dir, "judge_vs_human.json"))) if os.path.exists(os.path.join(a.run_dir, "judge_vs_human.json")) else None
        big = [m for m, s in pool["models"].items() if s.get("size_b", 0) >= 30]
        g = P2.gates(rep, jm, big_models=big, **cfg.get("gates", {}))
        md = P2.render_report_md(rep, g); open(os.path.join(a.run_dir, "REPORT.md"), "w").write(md); print(md)

if __name__ == "__main__":
    main()
