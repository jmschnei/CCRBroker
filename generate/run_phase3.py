"""CLI de la Fase 3 (DriftRoute).  Estructura de datos:

  data/drift/subset/{queries.*, split.json, subset_manifest.json}   <- congelado en `init`
  data/drift/s00/, s01/, ...  (configs, responses_*.jsonl, manifest_sXX.json, labels_*.jsonl opcionales)

Subcomandos:
  init      congela subconjunto + partición train/test (cluster)
  new-pool  genera el pool del siguiente snapshot aplicando eventos (y los registra)
  snapshot  ejecuta un snapshot (reanudable) + manifiesto + reprecio por throughput medido
  check     verifica comparabilidad entre snapshots
  report    deriva por config entre snapshots consecutivos (McNemar + Holm)
  eval      E4: evaluación prequential de routers (estático / reentrenado / perfil+online)
  simulate  ensayo en seco con datos sintéticos (sin GPU)
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import yaml

from routedata import drift as D
from routedata import pricing as PR
from routedata.schema import load_table, save_table
from routedata.stats import holm

from . import configroute as CR
from . import driftroute as DR
from .backends import MockBackend


def _mock_factory(pool):
    import hashlib

    names = list(pool["models"])

    def f(m, rev):
        base = 0.3 + 0.6 * (names.index(m) / max(1, len(names) - 1)) if m in names else 0.5
        delta = (
            int(hashlib.md5(str(rev).encode()).hexdigest()[:2], 16) % 3 - 1
        ) * 0.1  # la revisión altera la habilidad
        return MockBackend(m, skill=float(np.clip(base + delta, 0.05, 0.95)))

    return f


def _factory(kind, pool):
    if kind == "mock":
        return _mock_factory(pool)
    from .backends import VLLMBackend

    return lambda m, rev: VLLMBackend(m, rev, tensor_parallel_size=pool["models"][m].get("tp", 1))


def cmd_init(a, cfg):
    q = load_table(a.queries)
    ex = set(pd.read_json(a.exclude_ids, lines=True)["query_id"]) if a.exclude_ids else set()
    sub = DR.freeze_subset(q, cfg["subset_queries"], cfg["seed"], cfg["verifiable_only"], ex)
    tr, te = DR.split_train_test(sub, cfg["test_frac"], cfg["seed"])
    d = os.path.join(a.drift_dir, "subset")
    os.makedirs(d, exist_ok=True)
    save_table(sub, os.path.join(d, "queries"))
    json.dump(dict(train_ids=tr, test_ids=te), open(os.path.join(d, "split.json"), "w"))
    json.dump(
        dict(
            n=len(sub),
            queries_hash=DR.ids_hash(sub.query_id),
            seed=cfg["seed"],
            verifiable_only=cfg["verifiable_only"],
            n_train=len(tr),
            n_test=len(te),
        ),
        open(os.path.join(d, "subset_manifest.json"), "w"),
        indent=2,
    )
    print(f"subconjunto congelado: {len(sub)} consultas ({len(tr)} train / {len(te)} test)")


def cmd_new_pool(a, cfg):
    pool = yaml.safe_load(open(a.base_pool))
    p, ev = DR.apply_pool_events(
        pool,
        a.snapshot,
        add=a.add or [],
        retire=a.retire or [],
        revision=a.revision or [],
        price_factor=a.price_factor or [],
        new_batch=a.new_batch,
    )
    yaml.safe_dump(p, open(a.out, "w"), sort_keys=False)
    print(f"{len(ev)} evento(s) registrados en {a.out}")


def cmd_snapshot(a, cfg):
    pool = yaml.safe_load(open(a.pool))
    sub = load_table(os.path.join(a.drift_dir, "subset/queries"))
    q = sub
    if a.new_batch_file:
        nb = pd.read_json(a.new_batch_file, lines=True)
        nb["batch_id"] = a.batch_id
        q = pd.concat([sub, nb], ignore_index=True)
    models = {
        m: dict(revision=s["revision"], price_in=0.0, price_out=0.0, thinking=s["thinking"])
        for m, s in pool["models"].items()
    }
    cfgs = CR.build_config_grid(
        models,
        budgets=tuple(cfg["grid"]["budgets_tokens"]),
        efforts=tuple(cfg["grid"]["efforts"]),
        hardware=pool["gpu"]["type"],
        price_date=pool["gpu"]["price_date"],
    )
    out = os.path.join(a.drift_dir, a.snapshot)
    g = cfg["generation"]
    resp = CR.run(
        q,
        cfgs,
        _factory(a.backend or "vllm", pool),
        out,
        run_id=a.snapshot,
        snapshot_id=a.snapshot,
        seed=g["seed"],
    )
    tp = {m: s.get("tp", 1) for m, s in pool["models"].items()}
    cfgs = PR.reprice_configs(
        cfgs,
        PR.throughput_from_responses(resp, cfgs),
        pool["gpu"]["usd_per_hour"],
        tp,
        price_date=pool["gpu"]["price_date"],
    )
    for m, s in pool["models"].items():  # eventos de precio: factor sobre el precio derivado
        if s.get("price_factor"):
            cfgs.loc[cfgs.model == m, ["price_in", "price_out"]] *= s["price_factor"]
    save_table(cfgs, os.path.join(out, "configs"))
    man = DR.SnapshotManifest(
        snapshot_id=a.snapshot,
        date=a.date or pd.Timestamp.now().strftime("%Y-%m-%d"),
        models={m: s["revision"] for m, s in pool["models"].items()},
        engine="vllm" if (a.backend or "vllm") != "mock" else "mock",
        engine_version=DR.capture_environment().get("vllm", ""),
        hardware=pool["gpu"]["type"],
        price_table={
            r.model + "|" + r.config_id: dict(**{"in": r.price_in, "out": r.price_out}, source=r.price_source)
            for r in cfgs.itertuples()
        },
        events=pool.get("events", []),
        notes=a.note or "",
        queries_hash=DR.ids_hash(q.query_id),
        prompt_hash=DR.prompt_hash(),
        gen_params=g,
        judge=pool.get("judge_freeze", {}),
        grid=cfg["grid"],
        models_meta={
            m: dict(size_b=s.get("size_b", 8), thinking=s["thinking"]) for m, s in pool["models"].items()
        },
    )
    json.dump(
        dict(DR.asdict(man), env=DR.capture_environment()),
        open(os.path.join(out, f"manifest_{a.snapshot}.json"), "w"),
        indent=2,
    )
    print(f"snapshot {a.snapshot} listo: {len(resp)} respuestas")


def _manifests(drift_dir):
    ms = []
    for sid in sorted(
        d
        for d in os.listdir(drift_dir)
        if re.fullmatch(r"s\d+", d) and os.path.isdir(os.path.join(drift_dir, d))
    ):
        j = json.load(open(os.path.join(drift_dir, sid, f"manifest_{sid}.json")))
        j.pop("env", None)
        ms.append(DR.SnapshotManifest(**j))
    return ms


def cmd_check(a, cfg):
    w = DR.comparability_check(_manifests(a.drift_dir))
    print(
        "\n".join("⚠ " + x for x in w)
        if w
        else "✅ snapshots comparables (prompts, parámetros, juez y consultas coinciden)"
    )


def cmd_report(a, cfg):
    sids = [m.snapshot_id for m in _manifests(a.drift_dir)]
    md = ["# Informe de deriva por configuración\n"]
    for s0, s1 in zip(sids, sids[1:], strict=False):
        ra, rb = (
            CR.load_responses(os.path.join(a.drift_dir, s0)),
            CR.load_responses(os.path.join(a.drift_dir, s1)),
        )
        rep = DR.drift_report(ra.dropna(subset=["correct"]), rb.dropna(subset=["correct"]))
        if rep.empty:
            continue
        rep["p_holm"], rep["signif"] = holm(rep.p_mcnemar.values)
        ev = (
            ", ".join(
                e["type"] + ":" + e.get("model", "")
                for e in json.load(open(os.path.join(a.drift_dir, s1, f"manifest_{s1}.json")))["events"]
            )
            or "-"
        )
        md += [f"## {s0} → {s1}  (eventos: {ev})\n", rep.round(3).to_string(index=False), ""]
    p = os.path.join(a.drift_dir, "DRIFT_REPORT.md")
    open(p, "w").write("\n".join(md))
    print("escrito", p)


def _run_eval(snaps, train_ids, test_ids, cfg, drift_dir=None):
    df = D.prequential(snaps, set(train_ids), set(test_ids), tuple(cfg["fractions"]), seed=cfg["seed"])
    hs = D.h4_summary(df, cfg["h4"]["recovery_target"])
    h = cfg["h4"]
    ev = hs[hs.events.str.contains("add_model|new_revision|price_change|retire_model")]
    lines = [
        "# Resultados E4 (H4)\n",
        df.round(4).to_string(index=False),
        "\n## Resumen por evento\n",
        hs.round(3).to_string(index=False),
        "\n## Lectura de H4 (solo descriptiva; ver límites)\n",
    ]
    if len(ev):
        lines.append(
            f"- H4a (pérdida del estático ≥ {h['min_rel_loss']:.0%} de AIQ tras cambio de pool): se cumple en "
            f"{int((ev.rel_loss_static >= h['min_rel_loss']).sum())} de {len(ev)} eventos (mediana {ev.rel_loss_static.median():.1%})."
        )
        ok = ev.f_needed_profile_online.notna() & (ev.f_needed_profile_online <= h["max_traffic"])
        lines.append(
            f"- H4b (perfil+online recupera {h['recovery_target']:.0%} con ≤ {h['max_traffic']:.0%} del tráfico): se cumple en {int(ok.sum())} de {len(ev)} eventos."
        )
    lines.append(
        "- Límites: sin IC por evento (pocos snapshots; añade bootstrap sobre consultas de test antes de publicar); "
        "la 'recuperación' se mide respecto al reentrenado completo, que puede ser débil: mira siempre `ng_full` y `ng_static`."
    )
    return df, hs, "\n".join(lines)


def cmd_eval(a, cfg):
    sp = json.load(open(os.path.join(a.drift_dir, "subset/split.json")))
    snaps = DR.load_snaps(a.drift_dir)
    tests = set(sp["test_ids"]) | {
        q for s in snaps for q in s.qids[s.batch != "b0"]
    }  # lotes nuevos: solo test
    df, hs, md = _run_eval(snaps, sp["train_ids"], tests, cfg)
    df.to_csv(os.path.join(a.drift_dir, "e4_results.csv"), index=False)
    open(os.path.join(a.drift_dir, "E4_REPORT.md"), "w").write(md)
    print(md)


def cmd_simulate(a, cfg):
    snaps, q0 = D.simulate_drift(seed=cfg["seed"])
    tr, te = DR.split_train_test(pd.DataFrame({"query_id": q0}), 0.5, cfg["seed"])
    tests = set(te) | {q for s in snaps for q in s.qids[s.batch != "b0"]}
    _, _, md = _run_eval(snaps, tr, tests, cfg)
    print(md.split("## Resumen por evento")[1])


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("init", "new-pool", "snapshot", "check", "report", "eval", "simulate"):
        p = sub.add_parser(name)
        p.add_argument("--cfg", default="configs/driftroute.yaml")
        p.add_argument("--drift-dir", default="data/drift")
        for x in (
            "--queries",
            "--exclude-ids",
            "--base-pool",
            "--out",
            "--snapshot",
            "--pool",
            "--backend",
            "--note",
            "--date",
            "--new-batch-file",
            "--batch-id",
        ):
            p.add_argument(x)
        for x in ("--add", "--retire", "--revision", "--price-factor"):
            p.add_argument(x, action="append")
        p.add_argument("--new-batch", action="store_true")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.cfg))
    a.batch_id = a.batch_id or "b1"
    {
        "init": cmd_init,
        "new-pool": cmd_new_pool,
        "snapshot": cmd_snapshot,
        "check": cmd_check,
        "report": cmd_report,
        "eval": cmd_eval,
        "simulate": cmd_simulate,
    }[a.cmd](a, cfg)


if __name__ == "__main__":
    main()
