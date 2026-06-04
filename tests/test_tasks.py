from grpo_es.tasks.base import build_dataset
from grpo_es.tasks.registry import get_task_spec

import pytest


def test_toy_dataset_columns_and_answers():
    spec = get_task_spec("toy")
    ds = build_dataset(spec, max_samples=4)
    assert len(ds) == 4
    assert {"prompt", "answer", "words"} <= set(ds.column_names)
    for row in ds:
        words = [w.strip() for w in row["words"].split(",")]
        assert row["answer"] == "".join(w[-1] for w in words)
        assert row["words"] in row["prompt"]


def test_countdown_dataset_columns():
    spec = get_task_spec("countdown")
    ds = build_dataset(spec, max_samples=8)
    assert len(ds) == 8
    assert {"prompt", "answer", "numbers", "target"} <= set(ds.column_names)
    for row in ds:
        assert len(row["numbers"]) == 4
        assert row["answer"] == str(row["target"])
        assert str(row["target"]) in row["prompt"]


def test_loaders_are_seed_deterministic():
    spec = get_task_spec("countdown")
    a = build_dataset(spec, seed=7, max_samples=16)
    b = build_dataset(spec, seed=7, max_samples=16)
    c = build_dataset(spec, seed=8, max_samples=16)
    assert a["prompt"] == b["prompt"]
    assert a["prompt"] != c["prompt"]


def test_unknown_task_raises():
    with pytest.raises(KeyError, match="unknown task"):
        get_task_spec("definitely-not-a-task")
