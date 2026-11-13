"""Tests para el pipeline de datos, los adaptadores y la validación de datasets.

- `Pipeline.save` y la integridad referencial: siempre se ejecutan (datos sintéticos).
- Adaptadores sobre datos reales: solo si existe `data/raw/` en el repo (skip si no).
- `validate_all`: tanto con datos reales como contra un directorio vacío.

Ejecuta:  pytest tests/test_pipeline.py
"""
import json
from pathlib import Path

import pandas as pd
import pytest
import numpy as np

from routedata.pipeline import Pipeline
from routedata.schema import validate as validate_table
from routedata.validate import validate_all, validate_unified

ROOT = Path(__file__).resolve().parents[1]
HAS_RAW_DATA = (ROOT / "data" / "raw").is_dir()


def _small_dataset():
    queries = pd.DataFrame({"query_id": ["q1", "q2"], "text": ["a", "b"], "answer": ["x", "y"]})
    configs = pd.DataFrame({"config_id": ["c1", "c2"], "model": ["m1", "m2"]})
    responses = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q2"],
            "config_id": ["c1", "c2", "c1"],
            "run_id": ["r0", "r0", "r0"],
            "correct": [0.5, float("nan"), 0.9],
            "in_tokens": [10, 0, 20],
            "out_tokens": [5, 0, 10],
            "latency_s": [0.2, 0.0, 0.4],
            "cost_usd": [0.01, 0.0, 0.02],
            "label_source": ["test", "pending", "test"],
        }
    )
    return queries, configs, responses


def test_pipeline_save_writes_per_dataset_and_combined(tmp_path):
    out = tmp_path / "unified"
    pipeline = Pipeline(data_root=str(tmp_path / "root"))  # sin data/raw: solo probamos save()
    pipeline.save({"ds_a": _small_dataset()}, output_dir=str(out))

    expected = [
        "queries_ds_a.parquet", "configs_ds_a.parquet", "responses_ds_a.parquet",
        "queries_combined.parquet", "configs_combined.parquet", "responses_combined.parquet",
        "manifest.json",
    ]
    for name in expected:
        assert (out / name).exists(), f"falta {name}"

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["datasets"] == ["ds_a"]
    assert manifest["total_queries"] == 2 and manifest["total_configs"] == 2
    assert manifest["total_responses"] == 3
    assert manifest["files"]["queries"] == "queries_combined.parquet"

    # Cada tabla por dataset está etiquetada con su origen
    tagged = pd.read_parquet(out / "queries_ds_a.parquet", columns=["source_dataset"])
    assert set(tagged["source_dataset"]) == {"ds_a"}


def test_pipeline_save_referential_integrity(tmp_path):
    out = tmp_path / "unified"
    Pipeline(data_root=str(tmp_path / "root")).save({"ds_a": _small_dataset()}, output_dir=str(out))

    qids = set(pd.read_parquet(out / "queries_combined.parquet", columns=["query_id"])["query_id"])
    cids = set(pd.read_parquet(out / "configs_combined.parquet", columns=["config_id"])["config_id"])
    responses = pd.read_parquet(out / "responses_combined.parquet", columns=["query_id", "config_id"])
    assert responses["query_id"].isin(qids).all(), "query_ids huérfanos en responses"
    assert responses["config_id"].isin(cids).all(), "config_ids huérfanos en responses"
    report = validate_unified(out)
    assert report["valid"], report["errors"]
    assert report["counts"] == {"queries": 2, "configs": 2, "responses": 3}
    assert report["missing_labels"] == 1


def test_response_validation_allows_missing_correct_label():
    responses = pd.DataFrame({"query_id": ["q1", "q2"], "config_id": ["c1", "c1"],
                              "in_tokens": [1, 0], "out_tokens": [2, 0],
                              "correct": [1.0, float("nan")]})

    assert validate_table(responses, "responses") == []


def test_routerarena_queries_preserve_context_options_and_eval_only():
    from routedata.adapters.base import EvalOnlyError, assert_trainable
    from routedata.adapters.routerarena import to_queries

    source = pd.DataFrame({
        "Global Index": ["holdout-1"],
        "Context": ["Use the passage."],
        "Question": ["Which answer is correct?"],
        "Options": [np.array(["first", "second"])],
        "Answer": ["B"],
    })
    queries = to_queries(source)

    assert "Use the passage." in queries.loc[0, "text"]
    assert "1. first" in queries.loc[0, "text"]
    assert "2. second" in queries.loc[0, "text"]
    assert bool(queries.loc[0, "eval_only"])
    with pytest.raises(EvalOnlyError):
        assert_trainable(queries)


def test_validate_all_missing_data_reports_fail(tmp_path):
    reports = validate_all(tmp_path)  # directorio vacío: no hay datasets
    assert set(reports) == {"routerbench", "sprout", "llmrouterbench", "r2bench"}
    for report in reports.values():
        assert not report.valid
        assert report.errors


@pytest.mark.skipif(not HAS_RAW_DATA, reason="data/raw no presente en este checkout")
def test_routerbench_adapter_real_data():
    from routedata.adapters.routerbench import load, load_both

    for name in ("routerbench_0shot.pkl", "routerbench_5shot.pkl"):
        queries, configs, responses = load(str(ROOT / "data" / "raw" / "routerbench" / name))
        assert len(queries) > 0 and len(configs) > 0 and len(responses) > 0
        assert "query_id" in queries.columns and "text" in queries.columns
        assert "config_id" in configs.columns and "model" in configs.columns
        assert set(responses["query_id"]) <= set(queries["query_id"])
        assert set(responses["config_id"]) <= set(configs["config_id"])

    _, configs, _ = load_both(str(ROOT / "data" / "raw" / "routerbench"))
    assert configs["config_id"].is_unique
    assert len(configs) == 11


@pytest.mark.skipif(not HAS_RAW_DATA, reason="data/raw no presente en este checkout")
def test_sprout_adapter_real_data_schema_and_fk():
    from routedata.adapters.sprout import create_sprout_adapter

    queries, configs, responses = create_sprout_adapter(ROOT).load()

    # Regresión: SPROUT usaba (id, prompt, gold) y rompía la unificación
    assert {"query_id", "text", "answer"} <= set(queries.columns)
    assert {"config_id", "model"} <= set(configs.columns)
    assert len(queries) == queries["query_id"].nunique()
    assert set(responses["query_id"]) <= set(queries["query_id"])
    assert set(responses["config_id"]) <= set(configs["config_id"])
    assert responses["score"].between(0.0, 1.0).all()


@pytest.mark.skipif(not HAS_RAW_DATA, reason="data/raw no presente en este checkout")
def test_llmrouterbench_adapter_real_data():
    from routedata.adapters.llmrouterbench import LLMRouterBenchAdapter

    queries, configs, responses = LLMRouterBenchAdapter().load(str(ROOT / "data" / "raw" / "llmrouterbench"))
    assert len(queries) > 0 and len(configs) > 0 and len(responses) > 0
    assert "query_id" in queries.columns
    assert "prompt" in queries.columns
    assert queries["prompt"].str.len().gt(0).all()
    assert set(responses["query_id"]) <= set(queries["query_id"])
    assert set(responses["config_id"]) <= set(configs["config_id"])
    assert responses["out_tokens"].dropna().ge(0).all()


@pytest.mark.skipif(not HAS_RAW_DATA, reason="data/raw no presente en este checkout")
def test_r2bench_adapter_limited_load():
    from routedata.adapters.r2bench import create_r2bench_adapter

    queries, configs, responses = create_r2bench_adapter(ROOT, limit_files=2).load()
    assert len(responses) > 0
    assert {"query_id", "text", "answer"} <= set(queries.columns)
    assert {"config_id", "model"} <= set(configs.columns)
    assert set(responses["query_id"]) <= set(queries["query_id"])
    assert set(responses["config_id"]) <= set(configs["config_id"])


@pytest.mark.skipif(not HAS_RAW_DATA, reason="data/raw no presente en este checkout")
def test_validate_all_passes_on_local_data():
    reports = validate_all(ROOT / "data")
    summary = {name: report.valid for name, report in reports.items()}
    assert all(summary.values()), summary
