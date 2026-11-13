import numpy as np

from scripts.run_phase1_multidataset import (
    DATASET_SPEC,
    OPEN_WEIGHT_MODELS,
    _json_safe,
    split_dataset_by_prompt,
)


def test_official_pool_and_task_spec_sizes():
    assert len(OPEN_WEIGHT_MODELS) == 20
    assert len(DATASET_SPEC) == 15
    assert sum(item["expected_queries"] for item in DATASET_SPEC.values()) == 11_480


def test_split_keeps_duplicate_prompts_together_and_is_reproducible():
    query_ids = ["ds/test/0", "ds/test/1", "ds/test/2", "ds/test/3", "ds/test/4"]
    prompts = ["same prompt", "other", "same prompt", "third", "fourth"]

    train_a, test_a, counts_a = split_dataset_by_prompt(query_ids, prompts, 0.7, seed=42)
    train_b, test_b, counts_b = split_dataset_by_prompt(query_ids, prompts, 0.7, seed=42)

    assert np.array_equal(train_a, train_b)
    assert np.array_equal(test_a, test_b)
    assert counts_a == counts_b
    assert counts_a["unique_prompts"] == 4
    # Duplicate prompt rows must be on the same side of the split.
    assert (0 in train_a) == (2 in train_a)
    assert (0 in test_a) == (2 in test_a)
    assert set(train_a).isdisjoint(set(test_a))
    assert set(train_a) | set(test_a) == set(range(len(query_ids)))


def test_json_safe_converts_nonfinite_values_to_null():
    safe = _json_safe({"good": np.float64(0.5), "undefined": float("nan"), "values": [float("inf")]})

    assert safe == {"good": 0.5, "undefined": None, "values": [None]}
