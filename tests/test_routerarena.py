"""Puente con el evaluador oficial de RouterArena, sin red ni GPU (checkout falso + MockBackend)."""

import json

import pytest

from generate import routerarena as RA
from generate.backends import GenResult, MockBackend

MODELS = ["org/small", "org/big"]


@pytest.fixture
def arena(tmp_path):
    root = tmp_path / "RouterArena"
    (root / "dataset").mkdir(parents=True)
    (root / "model_cost").mkdir()
    rows = [{"global index": f"DS_{i}", "prompt_formatted": f"Question {i}?"} for i in range(809)]
    (root / "dataset" / "router_data_10.json").write_text(json.dumps(rows))
    (root / "universal_model_names.py").write_text('universal_names = [\n    "gpt-4o-mini",\n]\n')
    (root / "model_cost" / "model_profiles.yaml").write_text(
        'profiles:\n  "openai/gpt-4o-mini":\n    price: {input: 0.15, output: 0.6}\n'
    )
    return root


def test_load_prompts_checks_split_size(arena):
    prompts = RA.load_arena_prompts(arena, "sub_10")
    assert len(prompts) == 809 and prompts[0] == {"global_index": "DS_0", "prompt": "Question 0?"}

    data = json.loads((arena / "dataset" / "router_data_10.json").read_text())
    (arena / "dataset" / "router_data_10.json").write_text(json.dumps(data[:10]))
    with pytest.raises(ValueError, match="809"):
        RA.load_arena_prompts(arena, "sub_10")


def test_generated_result_matches_official_validity_rule():
    ok = RA.to_generated_result(GenResult("\\boxed{A}", 10, 5), "org/small")
    assert RA.is_valid_generation(ok)
    assert ok["token_usage"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "reasoning_tokens": 0,
        "cached_input_tokens": 0,
    }
    assert ok["model_used"] == "org/small"
    assert not RA.is_valid_generation(RA.to_generated_result(GenResult("", 10, 0), "org/small"))
    assert not RA.is_valid_generation(None)


def test_generate_is_resumable_and_predictions_follow_official_format(arena, tmp_path):
    prompts = RA.load_arena_prompts(arena)
    cache = tmp_path / "cache"
    calls = []

    def factory(m):
        calls.append(m)
        return MockBackend(m, skill=0.5)

    summary = RA.generate_candidates(prompts, MODELS, factory, cache, batch=200, log=lambda *_: None)
    assert summary == {m: 809 for m in MODELS} and calls == MODELS
    calls.clear()
    RA.generate_candidates(prompts, MODELS, factory, cache, log=lambda *_: None)
    assert calls == []  # todo en caché: no se instancia ningún backend

    choices = {p["global_index"]: MODELS[i % 2] for i, p in enumerate(prompts)}
    preds = RA.build_predictions(prompts, choices, MODELS, cache)
    main = [r for r in preds if not r["for_optimality"]]
    opt = [r for r in preds if r["for_optimality"]]
    assert len(main) == 809 and len(opt) == 809 * (len(MODELS) - 1)
    assert set(main[0]) == {
        "global index",
        "prompt",
        "prediction",
        "generated_result",
        "cost",
        "accuracy",
        "for_optimality",
    }
    assert all(r["prediction"] == choices[r["global index"]] for r in main)
    assert all(r["generated_result"]["model_used"] == r["prediction"] for r in preds)
    assert all(o["prediction"] != choices[o["global index"]] for o in opt)

    cfg, pred = RA.write_router_files(arena, "mi-router", MODELS, preds)
    assert json.loads(cfg.read_text())["pipeline_params"]["models"] == MODELS
    assert len(json.loads(pred.read_text())) == len(preds)


def test_build_predictions_rejects_incomplete_or_foreign_choices(arena, tmp_path):
    prompts = RA.load_arena_prompts(arena)
    with pytest.raises(ValueError, match="Faltan decisiones"):
        RA.build_predictions(prompts, {}, MODELS, tmp_path)
    with pytest.raises(ValueError, match="fuera del pool"):
        RA.build_predictions(prompts, {p["global_index"]: "other/model" for p in prompts}, MODELS, tmp_path)


def test_register_local_models_is_idempotent(arena):
    added = RA.register_local_models(arena, {"org/small": 8, "org/moe": 22})
    assert added == ["org/small", "org/moe"]
    assert RA.register_local_models(arena, {"org/small": 8, "org/moe": 22}) == []
    names = (arena / "universal_model_names.py").read_text()
    profiles = (arena / "model_cost" / "model_profiles.yaml").read_text()
    assert names.count('"org/small"') == 1 and '"gpt-4o-mini"' in names
    assert profiles.count('"org/moe":') == 1
    assert "local: {active_params_b: 22, bits: 16}" in profiles and "sync: false" in profiles
