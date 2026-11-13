"""CLI de la Fase 4 (AgentRoute).  Datos en data/agent/:
  tasks.json         entorno, semillas de tareas y partición train/test (por tarea)
  prices.json        {modelo: {"in","out" USD/Mtok, "prefill_tps","decode_tps"}}   (de `make-prices`)
  episodes.jsonl     episodios con políticas fijas (todas las tareas) y mixtas (train)
  stepdata.jsonl     etiquetas contrafactuales por (estado, modelo) en tareas de train
  eval_episodes.jsonl, eval_meta.json, AGENT_REPORT.md

Subcomandos: init · make-prices · collect · stepdata · evaluate · analyze · simulate
Modelos: pásalos de más barato a más caro (--models small,mid,big).
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from routedata import agent as A
from routedata.schema import load_table

from . import agentroute as AG
from .backends import MockBackend, OpenAICompatBackend
from .envs import get_env_factory


def _read(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def _append(p, rows):
    with open(p, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")


def _backends(a, models):
    if a.backend == "mock":
        sk = np.linspace(0.3, 0.95, max(2, len(models)))
        return {m: MockBackend(m, skill=float(sk[i]), verbosity=60) for i, m in enumerate(models)}
    ep = json.load(open(a.endpoints))  # {modelo: "http://host:puerto"}  (un `vllm serve` por modelo)
    return {m: OpenAICompatBackend(ep[m], m) for m in models}


def _ctx(a):
    d = a.data_dir
    T = json.load(open(os.path.join(d, "tasks.json")))
    fac = get_env_factory(T["env"], **T.get("env_kw", {}))
    pr = (
        json.load(open(os.path.join(d, "prices.json")))
        if os.path.exists(os.path.join(d, "prices.json"))
        else None
    )
    return d, T, fac, pr


def _episode(fac, be, policy, name, seed, max_steps, env_name):
    e = AG.run_episode(
        fac(),
        be,
        policy,
        f"{name}|t{seed}",
        name,
        seed=seed,
        max_steps=max_steps,
        task_id=f"t{seed}",
        env_name=env_name,
    )
    return dict(
        episode_id=e.episode_id,
        policy=e.policy,
        steps=e.steps,
        success=e.success,
        task_id=e.task_id,
        env_name=env_name,
    )


def _run_many(jobs, workers):
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return list(ex.map(lambda j: j(), jobs))


# ------------------------------------------------------------------ subcomandos
def cmd_init(a):
    os.makedirs(a.data_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    seeds = [int(x) for x in rng.permutation(10_000)[: a.n_tasks]]
    nt = int(round(a.n_tasks * a.test_frac))
    T = dict(
        env=a.env,
        env_kw=json.loads(a.env_kw or "{}"),
        tasks=seeds,
        test=seeds[:nt],
        train=seeds[nt:],
        seed=a.seed,
    )
    json.dump(T, open(os.path.join(a.data_dir, "tasks.json"), "w"))
    print(f"{len(T['train'])} tareas train / {len(T['test'])} test")


def cmd_make_prices(a):
    c = load_table(a.configs)
    g = c.groupby("model")[["price_in", "price_out"]].mean()
    pr = {m: {"in": float(r.price_in), "out": float(r.price_out)} for m, r in g.iterrows()}
    if a.tps:  # opcional: {modelo: {"prefill_tps":..., "decode_tps":...}} medido en las fases 2/3 (batch)
        for m, t in json.load(open(a.tps)).items():
            pr.setdefault(m, {}).update(t)
    json.dump(pr, open(os.path.join(a.data_dir, "prices.json"), "w"), indent=2)
    print(pr)


def cmd_collect(a):
    d, T, fac, pr = _ctx(a)
    models = a.models.split(",")
    be = _backends(a, models)
    p = os.path.join(d, "episodes.jsonl")
    done = {e["episode_id"] for e in _read(p)}
    jobs = []
    for sd in T["tasks"]:
        for m in models:
            n = f"fixed:{m}"
            if f"{n}|t{sd}" not in done:
                jobs.append(
                    lambda sd=sd, m=m, n=n: _episode(
                        fac, be, AG.fixed_policy(m), n, sd, a.max_steps, T["env"]
                    )
                )
    if "mixed" in (a.policies or "fixed,mixed"):
        for sd in T["train"]:
            specs = [(f"switch_k{k}", lambda k=k: AG.switch_every_k(models, k)) for k in (2, 4)] + [
                ("random", lambda sd=sd: AG.random_policy(models, sd))
            ]
            for n, pf in specs:
                if f"{n}|t{sd}" not in done:
                    jobs.append(
                        lambda sd=sd, n=n, pf=pf: _episode(fac, be, pf(), n, sd, a.max_steps, T["env"])
                    )
    res = _run_many(jobs, a.workers)
    _append(p, res)
    print(f"{len(res)} episodios nuevos")


def cmd_stepdata(a):
    d, T, fac, pr = _ctx(a)
    models = a.models.split(",")
    be = _backends(a, models)
    p = os.path.join(d, "stepdata.jsonl")
    done = {r["task_id"] for r in _read(p)}
    ref = a.reference_model or models[-1]
    todo = [sd for sd in T["train"] if f"t{sd}" not in done][: a.max_trajectories]

    def job(sd):
        return AG.collect_step_utilities(
            fac,
            sd,
            be,
            models,
            AG.epsilon_policy(ref, models, a.eps, sd),
            ref,
            pr,
            n_rollouts=a.n_samples,
            max_steps=a.max_steps,
            mode=a.mode,
            cache_mode=a.cache_mode,
            task_id=f"t{sd}",
        )

    rows = [r for rs in _run_many([lambda sd=sd: job(sd) for sd in todo], a.workers) for r in rs]
    _append(p, rows)
    print(f"{len(rows)} filas de {len(todo)} tareas")


def _fit(a, d, T, fac, pr, models, fit_tasks):
    ep = [
        e
        for e in _read(os.path.join(d, "episodes.jsonl"))
        if e["policy"].startswith("fixed:") and int(e["task_id"][1:]) in fit_tasks
    ]
    fr = A.episodes_to_frame(ep, pr, (a.cache_mode,))
    X, S, C = [], [], []
    for sd in sorted(fit_tasks):
        g = fr[fr.task_id == f"t{sd}"].set_index("policy")
        if not all(f"fixed:{m}" in g.index for m in models):
            continue
        X.append(A.featurize_text([fac().reset(sd)])[0])
        S.append([g.loc[f"fixed:{m}", "success"] for m in models])
        C.append([g.loc[f"fixed:{m}", f"cost_{a.cache_mode}"] for m in models])
    tr = A.TaskRouter().fit(np.array(X), np.array(S), np.array(C))
    rows = pd.DataFrame(
        [r for r in _read(os.path.join(d, "stepdata.jsonl")) if int(r["task_id"][1:]) in fit_tasks]
    )
    return tr, A.StepRouter(models).fit(rows)


def _policies(tr, sr, models, pr, lam, cache_mode, retention):
    return {
        "task_router": lambda: A.task_router_policy(tr, models, lam),
        "step_router_nopenalty": lambda: A.step_router_policy(
            sr, pr, lam, cache_mode, retention, penalty=False
        ),
        "step_router_penalty": lambda: A.step_router_policy(sr, pr, lam, cache_mode, retention, penalty=True),
    }


def cmd_evaluate(a):
    d, T, fac, pr = _ctx(a)
    models = a.models.split(",")
    be = _backends(a, models)
    lams = [float(x) for x in a.lams.split(",")]
    tr_tasks = list(T["train"])
    nv = max(1, len(tr_tasks) // 4)
    val, fit = tr_tasks[:nv], set(tr_tasks[nv:])
    tr, sr = _fit(a, d, T, fac, pr, models, fit)
    jobs = lambda tasks, pols, tag: [
        (lambda sd=sd, n=n, pf=pf: _episode(fac, be, pf(), f"{n}@{tag}", sd, a.max_steps, T["env"]))
        for sd in tasks
        for n, pf in pols.items()
    ]
    # 1) selección de λ en validación (tareas de train no usadas para ajustar): menor coste con éxito ≥ mejor fijo − 0.02
    vp = {
        f"{fam}@{lam:g}": pf
        for lam in lams
        for fam, pf in _policies(tr, sr, models, pr, lam, a.cache_mode, a.retention).items()
    }
    vres = _run_many(jobs(val, vp, "val"), a.workers)
    vdf = A.episodes_to_frame(vres, pr, (a.cache_mode,))
    vdf["fam"] = vdf.policy.str.split("@").str[0]
    vdf["lam"] = vdf.policy.str.split("@").str[1].astype(float)
    ve = [
        e
        for e in _read(os.path.join(d, "episodes.jsonl"))
        if e["policy"].startswith("fixed:") and int(e["task_id"][1:]) in set(val)
    ]
    best_fixed = (
        A.episodes_to_frame(ve, pr, (a.cache_mode,)).groupby("policy").success.mean().max() if ve else 1.0
    )
    chosen = {}
    for fam, g in vdf.groupby("fam"):
        s = g.groupby("lam").agg(success=("success", "mean"), cost=(f"cost_{a.cache_mode}", "mean"))
        ok = s[s.success >= best_fixed - 0.02]
        chosen[fam] = float(ok.cost.idxmin()) if len(ok) else float(s.success.idxmax())
    # 2) test
    tp = {fam: _policies(tr, sr, models, pr, chosen[fam], a.cache_mode, a.retention)[fam] for fam in chosen}
    tp["escalation"] = lambda: AG.escalation_policy(models[0], models[-1], 1)
    tp["switch_k2"] = lambda: AG.switch_every_k([models[0], models[-1]], 2)
    res = _run_many(jobs(T["test"], tp, "test"), a.workers)
    for r in res:
        r["policy"] = r["policy"].split("@")[0]
        r["episode_id"] = r["episode_id"]
    open(os.path.join(d, "eval_episodes.jsonl"), "w").close()
    _append(os.path.join(d, "eval_episodes.jsonl"), res)
    json.dump(
        dict(
            chosen_lambda=chosen,
            best_fixed_val=float(best_fixed),
            cache_mode=a.cache_mode,
            retention=a.retention,
            lams=lams,
            validation=vdf.groupby(["fam", "lam"])[["success", f"cost_{a.cache_mode}"]]
            .mean()
            .reset_index()
            .to_dict("records"),
        ),
        open(os.path.join(d, "eval_meta.json"), "w"),
        indent=2,
        default=float,
    )
    print("λ elegidos:", chosen)


def cache_model_check(eps, retention="keep"):
    """Compara los tokens cacheados que PREDICE el modelo de caché con los que reportó el servidor (si los expone)."""
    pred, rep = [], []
    for e in eps:
        bp = AG.billed_prefill(e["steps"], "prefix", retention)
        for s, b in zip(e["steps"], bp, strict=False):
            if s.get("cached_tokens") is not None:
                pred.append(s["in_tokens"] - b)
                rep.append(s["cached_tokens"])
    if not pred:
        return None
    pred, rep = np.array(pred, float), np.array(rep, float)
    return dict(n=len(pred), mean_abs_err_frac=float(np.abs(pred - rep).sum() / max(rep.sum(), 1)))


def cmd_analyze(a):
    d, T, fac, pr = _ctx(a)
    test = {f"t{s}" for s in T["test"]}
    meta = json.load(open(os.path.join(d, "eval_meta.json")))
    eps = [
        e
        for e in _read(os.path.join(d, "episodes.jsonl")) + _read(os.path.join(d, "eval_episodes.jsonl"))
        if e["task_id"] in test
    ]
    cms = ("none", "prefix")
    df = A.episodes_to_frame(eps, pr, cms, meta["retention"])
    df = df.drop_duplicates(["policy", "task_id"])
    summ = A.summarize(df, cms)
    fixed = df[df.policy.str.startswith("fixed:")]
    hd = A.headroom(fixed, "prefix")
    md = [
        "# Informe AgentRoute (H5)\n",
        f"λ elegidos en validación: {meta['chosen_lambda']} (modo de caché del router: {meta['cache_mode']}, retención: {meta['retention']})\n",
        "## Resumen por política (tareas de test)\n",
        summ.round(4).to_string(index=False),
        "\n## Margen para enrutar (headroom)\n",
        hd["per_model"].to_string(),
        f"\nOráculo por tarea: éxito {hd['oracle_task_success']:.3f} a coste {hd['oracle_task_cost']:.5f}; mejor fijo: {hd['best_fixed']} (gap de éxito {hd['gap_success']:+.3f}, razón de coste {hd['cost_ratio_oracle_vs_best']:.2f})\n",
    ]
    for cm in cms:
        cmp_ = A.compare_policies(df, "task_router", ["step_router_penalty", "step_router_nopenalty"], cm)
        md += [
            f"## H5 · comparación pareada por tarea (cache_mode={cm}), base = task_router\n",
            cmp_.round(4).to_string(index=False),
            "",
        ]
    chk = cache_model_check(_read(os.path.join(d, "episodes.jsonl")))
    md.append(
        f"## Validación del modelo de caché\n{chk if chk else 'el servidor no reportó cached_tokens: no se pudo contrastar'}"
    )
    md.append(
        "\n## Límites\n- Un solo entorno y pocas tareas → IC anchos; no generalices.\n- Los routers son referencias simples (kNN/Ridge); el coste de GPU depende de la concurrencia real.\n- Con feedback LOCAL el router ignora consecuencias a largo plazo (ver mode=rollout)."
    )
    open(os.path.join(d, "AGENT_REPORT.md"), "w").write("\n".join(md))
    print("\n".join(md))


def cmd_simulate(a):
    import tempfile

    d = a.data_dir or tempfile.mkdtemp()
    base = ["--data-dir", d, "--backend", "mock", "--models", "small,mid,big", "--workers", "4"]
    prices = {
        "small": {"in": 0.05, "out": 0.1},
        "mid": {"in": 0.15, "out": 0.3},
        "big": {"in": 2.0, "out": 6.0},
    }
    main(
        [
            "init",
            "--data-dir",
            d,
            "--env",
            "synthetic",
            "--env-kw",
            '{"horizon": 8, "n_hard": 3, "obs_chars": 2000}',
            "--n-tasks",
            str(a.n_tasks or 60),
            "--test-frac",
            "0.5",
        ]
    )
    json.dump(prices, open(os.path.join(d, "prices.json"), "w"))
    main(["collect", *base])
    main(["stepdata", *base, "--n-samples", "3", "--eps", "0.4"])
    main(["evaluate", *base, "--lams", "10,30,100,300"])
    main(["analyze", *base])


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("init", "make-prices", "collect", "stepdata", "evaluate", "analyze", "simulate"):
        p = sub.add_parser(n)
        p.add_argument("--data-dir", default=None if n == "simulate" else "data/agent")
        for x in (
            "--env",
            "--env-kw",
            "--configs",
            "--tps",
            "--models",
            "--backend",
            "--endpoints",
            "--policies",
            "--reference-model",
            "--mode",
            "--cache-mode",
            "--retention",
            "--lams",
        ):
            p.add_argument(x)
        for x, t, dv in (
            ("--n-tasks", int, None),
            ("--seed", int, 0),
            ("--max-steps", int, 30),
            ("--workers", int, 4),
            ("--max-trajectories", int, 10**9),
            ("--n-samples", int, 3),
        ):
            p.add_argument(x, type=t, default=dv)
        p.add_argument("--test-frac", type=float, default=0.4)
        p.add_argument("--eps", type=float, default=0.4)
        p.set_defaults(
            env="synthetic",
            backend="mock",
            mode="local",
            cache_mode="prefix",
            retention="keep",
            lams="10,30,100,300,1000",
        )
    a = ap.parse_args(argv)
    {
        "init": cmd_init,
        "make-prices": cmd_make_prices,
        "collect": cmd_collect,
        "stepdata": cmd_stepdata,
        "evaluate": cmd_evaluate,
        "analyze": cmd_analyze,
        "simulate": cmd_simulate,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
