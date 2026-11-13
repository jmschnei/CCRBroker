import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd

from generate import build_queries as BQ
from generate import configroute as CR
from generate import multilingual as ML
from generate import phase2 as P2
from generate.backends import GenResult, MockBackend
from routedata.schema import validate


class JudgeMock:
    name = "judge"

    def generate(self, reqs):
        return [GenResult('{"score": 0.8}', 10, 5) for _ in reqs]


def test_build_queries():
    mmlu = pd.DataFrame(
        {
            "question": ["2+2?", "Capital?"],
            "options": [["3", "4"], ["Rom", "Paris"]],
            "answer": ["B", "B"],
            "category": ["math", "geo"],
        }
    )
    q = BQ.normalize(mmlu, "mmlu_pro", BQ.SOURCES["mmlu_pro"])
    assert validate(q, "queries") == [] and q.answer.tolist() == ["B", "B"]
    gp = pd.DataFrame(
        {
            "Question": ["q"] * 20,
            "Correct Answer": ["RIGHT"] * 20,
            "Incorrect Answer 1": ["w1"] * 20,
            "Incorrect Answer 2": ["w2"] * 20,
            "Incorrect Answer 3": ["w3"] * 20,
        }
    )
    g = BQ.normalize(gp, "gpqa_diamond", BQ.SOURCES["gpqa_diamond"], seed=3)
    for _, r in g.iterrows():  # la letra dorada apunta a la opción correcta
        assert json.loads(r.options)["ABCD".index(r.answer)] == "RIGHT"
    he = pd.DataFrame(
        {
            "prompt": ["def f(x):\n    pass"],
            "test": ["def check(f):\n    assert f(1)==2"],
            "entry_point": ["f"],
        }
    )
    c = BQ.normalize(he, "humaneval", BQ.SOURCES["humaneval"])
    assert "check(f)" in json.loads(c.meta[0])["tests"]
    from generate.verifiers import verify

    ok, src = verify(
        "code", "```python\ndef f(x):\n    return x+1\n```", "", tests=json.loads(c.meta[0])["tests"]
    )
    assert ok == 1.0
    Q = np.array([[1, 1, 1], [0, 0, 0], [1, 0, 1.0]])
    d = BQ.assign_empirical_difficulty(q.assign(x=0).iloc[:1].reindex([0, 0, 0]).reset_index(drop=True), Q)
    assert d.difficulty.tolist() == ["easy", "hard", "medium"]


def test_sample_queries():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "query_id": [f"q{i}" for i in range(500)],
            "domain": rng.choice(["a", "b"], 500),
            "difficulty": rng.choice(["e", "h"], 500),
        }
    )
    out = CR.sample_queries({"s": df}, {"s": 100}, seed=1)
    assert len(out) <= 100 and {"domain", "difficulty"} <= set(out.columns) and out.query_id.is_unique
    # la mezcla de dominios se mantiene aproximadamente
    assert abs((out.domain == "a").mean() - (df.domain == "a").mean()) < 0.12


def test_grid_no_duplicate_configs():
    models = {
        "t": dict(revision="r", price_in=0, price_out=0, thinking=True),
        "n": dict(revision="r", price_in=0, price_out=0, thinking=False),
    }
    g = CR.build_config_grid(models, budgets=(100, 500, None), efforts=("nothink", "think_low", "think_mid"))
    assert g.config_id.is_unique and len(g) == (3 + 2) + 3  # t: 3 nothink + 2 think ; n: 3 nothink
    assert g[g.effort.str.startswith("think")].budget.isna().all()


def test_multilingual():
    q = BQ.normalize(
        pd.DataFrame(
            {
                "question": ["What is 12 plus 5?"],
                "options": [["16", "17"]],
                "answer": ["B"],
                "category": ["m"],
            }
        ),
        "mmlu_pro",
        BQ.SOURCES["mmlu_pro"],
    )
    good = (
        lambda p: "¿Cuánto es 12 más 5?" if "12" in p else ("16" if "16" in p else "17" if "17" in p else "x")
    )
    t, ctrl = ML.translate_queries(q, "es", good, back_fn=lambda p: "What is 12 plus 5?")
    assert t.language[0] == "es" and not ctrl.needs_review[0] and t.answer[0] == "B"
    bad = lambda p: "¿Cuánto es 13 más 5?"  # cambia un número -> a revisión humana
    _, ctrl2 = ML.translate_queries(q, "es", bad)
    assert ctrl2.needs_review[0]


def test_pilot_judge_audit():
    rng = np.random.default_rng(0)
    n = 60
    q = pd.DataFrame(
        {
            "query_id": [f"q{i}" for i in range(n)],
            "text": [f"t{i}" for i in range(n)],
            "options": [json.dumps(["a", "b", "c", "d"]) if i < 30 else "" for i in range(n)],
            "answer": ["A" if i < 30 else "ref" for i in range(n)],
            "domain": ["math" if i % 2 else "sci" for i in range(n)],
            "source_dataset": "s",
            "language": "en",
            "meta": [
                json.dumps({"task_type": "mc", "difficulty": float(rng.random())})
                if i < 30
                else json.dumps({"task_type": "open", "difficulty": float(rng.random())})
                for i in range(n)
            ],
        }
    )
    models = {
        "small": dict(revision="r", price_in=0.1, price_out=0.2, thinking=False),
        "big": dict(revision="r", price_in=1, price_out=2, thinking=True),
    }
    cfgs = CR.build_config_grid(models, budgets=(100, None), efforts=("nothink", "think_low"))
    skills = {"small": 0.4, "big": 0.85}
    with tempfile.TemporaryDirectory() as d:
        resp = CR.run(q, cfgs, lambda m, r: MockBackend(m, skills[m]), d, batch=20, log=lambda *_: None)
        assert (resp.label_source == "pending_judge").sum() == (n - 30) * len(cfgs)  # abiertas -> juez
        rep = P2.analyze_pilot(q, cfgs, resp, n_full=8000, tp_by_model={"big": 4})
        assert rep.gpu_hours_full.min() > 0 and rep.decode_tps_agg.notna().all()
        lab = P2.apply_judges(q, resp, [JudgeMock(), JudgeMock()])
        assert (lab.correct == 0.8).all()
        full = P2.overlay_labels(resp, lab)
        assert full.label_source.eq("pending_judge").sum() == 0
        sheet = P2.export_audit(full, q, lab, n=40, out_dir=os.path.join(d, "audit"))
        assert "model" not in "".join(sheet.columns) and "config_id" not in sheet.columns  # ciega
        s = pd.read_csv(os.path.join(d, "audit/audit_sheet.csv"))
        s["human_score_a"] = 1.0
        s["human_score_b"] = 1.0
        s.to_csv(os.path.join(d, "audit/done.csv"), index=False)
        met, merged = P2.import_audit(
            os.path.join(d, "audit/done.csv"), os.path.join(d, "audit/audit_key_PRIVATE.csv")
        )
        assert met["n"] > 0 and "kappa_annotators" in met and len(merged) == met["n"]
        hl = P2.human_label_layer(merged)
        assert (hl.label_source == "human").all()
        g = P2.gates(rep, {"pearson": 0.85, "kappa": 0.75}, big_models=("big",))
        assert g["juez_vs_humano"][0]
        assert "GPU-horas" in P2.render_report_md(rep, g)


if __name__ == "__main__":
    for t in (
        test_build_queries,
        test_sample_queries,
        test_grid_no_duplicate_configs,
        test_multilingual,
        test_pilot_judge_audit,
    ):
        t()
        print("OK ", t.__name__)
