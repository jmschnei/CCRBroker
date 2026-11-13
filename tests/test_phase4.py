import contextlib
import io
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd

from generate import agentroute as AG
from generate import run_phase4 as P4
from generate.backends import GenRequest, MockBackend, OpenAICompatBackend
from generate.envs import get_env_factory
from routedata import agent as A
from routedata.schema import save_table


def _steps(spec):
    return [
        dict(model=m, in_tokens=i, out_tokens=o, latency_s=0.1, switched=False, obs_error=False)
        for m, i, o in spec
    ]


def test_cache_cost_model():
    st = _steps([("A", 100, 10), ("A", 130, 10), ("B", 170, 10), ("A", 210, 10)])
    assert AG.billed_prefill(st, "none") == [100, 130, 170, 210]
    assert AG.billed_prefill(st, "prefix", "keep") == [
        100,
        20,
        170,
        70,
    ]  # A conserva su caché; B arranca en frío
    assert AG.billed_prefill(st, "prefix", "evict") == [100, 20, 170, 210]  # tras B, A pierde la caché
    assert (
        AG.candidate_prefill(st[:3], "A", 210, "prefix", "keep") == 70
        and AG.candidate_prefill(st[:3], "B", 210, "prefix", "keep") == 30
    )
    price = {"A": {"in": 1.0, "out": 2.0}, "B": {"in": 3.0, "out": 4.0}}
    c_none, _ = AG.steps_cost(st, price, "none")
    c_pre, _ = AG.steps_cost(st, price, "prefix")
    assert c_pre < c_none
    # legado: precios como tupla y Episode.totals
    ep = AG.Episode("e", "p", st, 1.0)
    assert ep.totals({"A": (1.0, 2.0), "B": (3.0, 4.0)})["switches"] == 0


def test_replay_clone_and_state():
    fac = get_env_factory("synthetic", horizon=6, n_hard=2, obs_chars=200)
    e = fac()
    e.reset(7)
    [e.step(a) for a in ["ok", "bad", "ok"]]
    c = e.clone()
    assert (c.i, c.history) == (e.i, e.history) and c.step("ok") == e.step("ok")
    assert AG.check_deterministic(fac, 7, ["ok", "bad", "ok", "ok"]) == 1.0
    be = {"a": MockBackend("a", 0.9, 60)}
    st = AG.AgentState(fac(), be, 3)
    st.step("a")
    cl = st.clone()
    cl.step("a")
    assert len(st.steps) == 1 and len(cl.steps) == 2 and len(st.msgs) < len(cl.msgs)


def test_probes_and_rollouts():
    fac = get_env_factory("synthetic", horizon=6, n_hard=2, obs_chars=400)
    models = ["small", "big"]
    be = {"small": MockBackend("small", 0.3, 60), "big": MockBackend("big", 0.95, 60)}
    price = {m: {"in": 1.0, "out": 2.0} for m in models}
    rows = pd.DataFrame(
        sum(
            (
                AG.collect_step_utilities(
                    fac,
                    s,
                    be,
                    models,
                    AG.epsilon_policy("big", models, 0.3, s),
                    "big",
                    price,
                    n_rollouts=3,
                    task_id=f"t{s}",
                )
                for s in range(6)
            ),
            [],
        )
    )
    rows["hard"] = rows.obs.str.contains("analyze")
    u = rows.groupby(["hard", "model"]).utility.mean()
    assert (
        u[(True, "big")] > u[(True, "small")] + 0.5
        and abs(u[(False, "big")] - u[(False, "small")]) < 0.3
        and (rows.cost_after == 0).all()
    )
    rr = pd.DataFrame(
        AG.collect_step_utilities(
            fac, 1, be, models, AG.fixed_policy("big"), "big", price, n_rollouts=2, mode="rollout"
        )
    )
    assert rr.utility.between(0, 1).all() and (rr.cost_after > 0).any()


def test_step_router_penalty_regime():
    fac = get_env_factory("synthetic", horizon=8, n_hard=3, obs_chars=2000)
    models = ["small", "mid", "big"]
    be = {
        "small": MockBackend("small", 0.3, 60),
        "mid": MockBackend("mid", 0.6, 60),
        "big": MockBackend("big", 0.95, 60),
    }
    price = {
        "small": {"in": 0.05, "out": 0.1},
        "mid": {"in": 0.15, "out": 0.3},
        "big": {"in": 2.0, "out": 6.0},
    }
    rows = pd.DataFrame(
        sum(
            (
                AG.collect_step_utilities(
                    fac,
                    s,
                    be,
                    models,
                    AG.epsilon_policy("big", models, 0.4, s),
                    "big",
                    price,
                    n_rollouts=3,
                    task_id=f"t{s}",
                )
                for s in range(30)
            ),
            [],
        )
    )
    sr = A.StepRouter(models).fit(rows)
    pols = {
        "fixed:big": lambda: AG.fixed_policy("big"),
        "nopen": lambda: A.step_router_policy(sr, price, 300, "prefix", "keep", False),
        "pen": lambda: A.step_router_policy(sr, price, 300, "prefix", "keep", True),
    }
    eps = []
    for sd in range(100, 130):
        for n, pf in pols.items():
            e = AG.run_episode(fac(), be, pf(), f"{n}-{sd}", n, seed=sd, task_id=f"t{sd}")
            eps.append(
                dict(
                    episode_id=e.episode_id,
                    policy=e.policy,
                    steps=e.steps,
                    success=e.success,
                    task_id=e.task_id,
                )
            )
    s = A.summarize(A.episodes_to_frame(eps, price)).set_index("policy")
    assert s.loc["nopen", "switches"] > s.loc["pen", "switches"]  # ignorar la caché → cambia de modelo más
    r = lambda p: s.loc[p, "cost_prefix"] / s.loc[p, "cost_none"]
    assert r("nopen") > r("fixed:big")  # y la caché le beneficia menos que al fijo


def test_compare_headroom_summary():
    rows = []
    for t in range(40):
        for pol, succ, cost in (("task_router", int(t % 4 != 0), 1.0), ("step", 1, 0.8)):
            rows.append(
                dict(
                    task_id=f"t{t}",
                    policy=pol,
                    success=float(succ),
                    steps=5,
                    switches=0,
                    cost_none=cost,
                    lat_none=1.0,
                    cost_prefix=cost / 2,
                    lat_prefix=1.0,
                )
            )
    df = pd.DataFrame(rows)
    c = A.compare_policies(df, "task_router", ["step"], "prefix")
    assert c.d_success.iloc[0] > 0.2 and c.d_cost.iloc[0] < 0 and c.p_mcnemar.iloc[0] < 0.05
    fx = pd.DataFrame(
        [
            dict(task_id=f"t{i}", policy=p, success=s, cost_prefix=cst)
            for i, (a, b) in enumerate([(1, 0), (1, 0), (0, 1), (0, 1)])
            for p, s, cst in (("fixed:A", a, 1.0), ("fixed:B", b, 2.0))
        ]
    )
    h = A.headroom(fx)
    assert h["oracle_task_success"] == 1.0 and abs(h["gap_success"] - 0.5) < 1e-9


class _H(BaseHTTPRequestHandler):
    bodies = []

    def do_POST(self):
        _H.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        out = json.dumps(
            {
                "choices": [{"message": {"content": "hola"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 7,
                    "prompt_tokens_details": {"cached_tokens": 64},
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_openai_compat_backend():
    srv = HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        b = OpenAICompatBackend(f"http://127.0.0.1:{srv.server_port}", "m")
        g = b.generate([GenRequest([{"role": "user", "content": "x"}], enable_thinking=False, seed=3)])[0]
        assert (g.text, g.in_tokens, g.out_tokens, g.finish_reason, g.cached_tokens) == (
            "hola",
            100,
            7,
            "stop",
            64,
        )
        body = _H.bodies[-1]
        assert (
            body["chat_template_kwargs"] == {"enable_thinking": False}
            and body["seed"] == 3
            and body["model"] == "m"
        )
    finally:
        srv.shutdown()


def test_cli_simulate_and_prices():
    with tempfile.TemporaryDirectory() as t:
        q = io.StringIO()
        with contextlib.redirect_stdout(q):
            P4.main(["simulate", "--data-dir", os.path.join(t, "ag"), "--n-tasks", "24"])
        out = q.getvalue()
        assert "H5" in out and "λ elegidos" in out
        for f in (
            "tasks.json",
            "episodes.jsonl",
            "stepdata.jsonl",
            "eval_episodes.jsonl",
            "eval_meta.json",
            "AGENT_REPORT.md",
        ):
            assert os.path.exists(os.path.join(t, "ag", f)), f
        save_table(
            pd.DataFrame(
                {"model": ["a", "a", "b"], "price_in": [1.0, 3.0, 5.0], "price_out": [2.0, 4.0, 6.0]}
            ),
            os.path.join(t, "cfg"),
        )
        os.makedirs(os.path.join(t, "p"), exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            P4.main(["make-prices", "--data-dir", os.path.join(t, "p"), "--configs", os.path.join(t, "cfg")])
        pr = json.load(open(os.path.join(t, "p/prices.json")))
        assert pr["a"]["in"] == 2.0 and pr["b"]["out"] == 6.0


if __name__ == "__main__":
    for f in (
        test_cache_cost_model,
        test_replay_clone_and_state,
        test_probes_and_rollouts,
        test_step_router_penalty_regime,
        test_compare_headroom_summary,
        test_openai_compat_backend,
        test_cli_simulate_and_prices,
    ):
        f()
        print("OK ", f.__name__)
