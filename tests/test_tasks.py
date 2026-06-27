from grpo_es.rewards.countdown_rubric import validate_expression
from grpo_es.tasks.base import build_dataset, build_eval_dataset
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
        # `answer` is now a CONSTRUCTED solution expression (not the target
        # string); it must validate as an exact solve over the row's operands.
        is_exact, _ = validate_expression(row["answer"], row["numbers"], row["target"])
        assert is_exact
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


# --- benchmark tasks (download a few rows from the HF hub) -------------------


def test_gsm8k_rows_have_extracted_answers():
    spec = get_task_spec("gsm8k")
    ds = build_dataset(spec, max_samples=2)
    assert len(ds) == 2
    assert {"prompt", "answer", "question"} <= set(ds.column_names)
    for row in ds:
        # The gold field upstream is full reasoning ending "#### N" — the
        # loader must keep only N.
        assert "####" not in row["answer"]
        assert "Problem:" in row["prompt"]
        assert "\\boxed{final_number}" in row["prompt"]


def test_mmlu_pro_rows_are_lettered_mcq():
    spec = get_task_spec("mmlu_pro")
    # "test" maps to the upstream validation split (70 rows, fast).
    ds = build_dataset(spec, split="test", max_samples=2)
    assert len(ds) == 2
    assert {"prompt", "answer", "question", "category"} <= set(ds.column_names)
    for row in ds:
        assert row["answer"] in "ABCDEFGHIJ"
        assert "\nA. " in row["prompt"]
        assert "\\boxed{X}" in row["prompt"]


# --- held-out eval slices ----------------------------------------------------


def test_eval_slice_matches_offset_arithmetic():
    # [offset : offset+size] of a seeded load must equal loading offset+size
    # rows and dropping the first offset — that identity is what
    # build_eval_dataset relies on.
    spec = get_task_spec("countdown")
    full = build_dataset(spec, seed=5, max_samples=10)
    sliced = build_eval_dataset(spec, seed=5, offset=4, size=6)
    assert sliced["prompt"] == full["prompt"][4:10]


def test_generated_tasks_hold_out_a_fresh_seed():
    spec = get_task_spec("countdown")
    train = build_dataset(spec, seed=0, max_samples=16)
    heldout = build_eval_dataset(spec, size=16)
    assert heldout["prompt"] == build_dataset(spec, seed=999, max_samples=16)["prompt"]
    assert heldout["prompt"] != train["prompt"]
