"""Hub-env adapter: worker protocol, prompt flattening, dynamic registries.

Everything except the last test runs under the normal training env — the
worker imports verifiers lazily, so the protocol tests can drive it with
``sys.executable``. The end-to-end test needs a real ``.venv-prime`` with
``primeintellect/gsm8k`` installed and skips cleanly without one.
"""

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest
from datasets import Dataset

from grpo_es.config.run_config import task_arg
from grpo_es.rewards.registry import DYNAMIC_RUBRICS, get_rubric, register_rubric
from grpo_es.tasks.base import TaskSpec
from grpo_es.tasks.from_env import (
    HubEnvClient,
    RemoteEnvRubric,
    default_prompt_transform,
    drop_eval_window,
    hub_python,
    messages_text,
)

WORKER = Path(__file__).resolve().parent.parent / "scripts" / "prime_env_worker.py"


# --- worker protocol ----------------------------------------------------------


@pytest.fixture
def worker():
    proc = subprocess.Popen(
        [sys.executable, str(WORKER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    def rpc(**req):
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        return json.loads(proc.stdout.readline())

    yield rpc
    proc.kill()


def test_worker_ping(worker):
    resp = worker(op="ping")
    assert resp["ok"] and resp["op"] == "pong"


def test_worker_unknown_op_is_error_not_death(worker):
    resp = worker(op="nope")
    assert not resp["ok"] and "unknown op" in resp["error"]
    assert worker(op="ping")["ok"]  # still alive after the error


def test_worker_score_before_load_errors(worker):
    resp = worker(op="score", env_id="ghost", rollouts=[])
    assert not resp["ok"] and "not loaded" in resp["error"]


# --- prompt flattening --------------------------------------------------------


def test_transform_string_prompt_gets_system():
    assert default_prompt_transform("Q?", "SYS") == "SYS\n\nQ?"


def test_transform_chat_without_system_gets_prepend():
    msgs = [{"role": "user", "content": "Q?"}]
    assert default_prompt_transform(msgs, "SYS") == "SYS\n\nQ?"


def test_transform_chat_with_baked_system_not_duplicated():
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "Q?"}]
    assert default_prompt_transform(msgs, "SYS") == "SYS\n\nQ?"


def test_messages_text_roundtrip():
    assert messages_text([{"role": "assistant", "content": "hi"}]) == "hi"
    assert messages_text("raw") == "raw"
    assert messages_text(None) == ""


# --- CLI validation + dynamic registries --------------------------------------


def test_task_arg_accepts_env_prefix_rejects_unknown():
    assert task_arg("env:owner/thing") == "env:owner/thing"
    assert task_arg("gsm8k") == "gsm8k"
    with pytest.raises(argparse.ArgumentTypeError):
        task_arg("not-a-task")


def test_dynamic_rubric_registry_fallback():
    sentinel = object()
    register_rubric("test:dyn", lambda: sentinel)
    try:
        assert get_rubric("test:dyn") is sentinel
        with pytest.raises(KeyError):
            get_rubric("test:unregistered")
    finally:
        DYNAMIC_RUBRICS.pop("test:dyn", None)


# --- single-pool holdout carving + reward_metric --------------------------------


def _if_like_spec(**overrides) -> TaskSpec:
    fields = dict(
        name="t",
        rubric="t",
        system_prompt="",
        build_prompt=lambda row: row["prompt"],
        eval_seed=3,
        eval_offset=2,
        eval_size=5,
    )
    fields.update(overrides)
    return TaskSpec(**fields)


def test_drop_eval_window_is_disjoint_from_eval_slice():
    ds = Dataset.from_list([{"id": i} for i in range(20)])
    spec = _if_like_spec()
    # What build_eval_dataset extracts: shuffle(eval_seed)[offset:offset+size].
    held_out = ds.shuffle(seed=spec.eval_seed).select(
        range(spec.eval_offset, spec.eval_offset + spec.eval_size)
    )
    train_pool = drop_eval_window(ds, spec)
    assert len(train_pool) == 15
    assert set(train_pool["id"]).isdisjoint(held_out["id"])
    assert set(train_pool["id"]) | set(held_out["id"]) == set(range(20))


class _StubClient:
    def score(self, env_id, rollouts):
        return [0.0] * len(rollouts), [{"some_rate": 0.4}] * len(rollouts)


def _scored_state(rubric: RemoteEnvRubric) -> dict:
    state = {"prompt": "p", "completion": "c", "answer": "", "info": {}}
    return asyncio.run(rubric.score_rollout(state))


def test_reward_metric_promotes_env_metric(monkeypatch):
    monkeypatch.setattr(HubEnvClient, "_shared", _StubClient())
    assert _scored_state(RemoteEnvRubric("e"))["reward"] == 0.0
    assert (
        _scored_state(RemoteEnvRubric("e", reward_metric="some_rate"))["reward"]
        == 0.4
    )
    # Missing metric falls back to the env reward instead of crashing.
    assert (
        _scored_state(RemoteEnvRubric("e", reward_metric="ghost"))["reward"] == 0.0
    )


# --- end-to-end (needs .venv-prime with primeintellect/gsm8k) ------------------


def _venv_ready() -> bool:
    try:
        hub_python()
        return True
    except FileNotFoundError:
        return False


@pytest.mark.skipif(
    not _venv_ready(), reason="no .venv-prime (scripts/setup_prime_venv.sh)"
)
def test_gsm8k_env_end_to_end():
    from grpo_es.rewards.trl_bridge import rubric_reward_func
    from grpo_es.tasks.base import build_dataset
    from grpo_es.tasks.registry import get_task_spec

    spec = get_task_spec("env:primeintellect/gsm8k")
    assert spec.rubric == "env:primeintellect/gsm8k"
    assert spec.format_scaffold is False

    ds = build_dataset(spec, split="train", seed=0, max_samples=2)
    assert len(ds) == 2
    assert {"prompt", "answer"} <= set(ds.column_names)
    row = ds[0]
    # The env bakes its system prompt into the chat; flattening must not
    # have doubled it.
    if spec.system_prompt:
        assert row["prompt"].count(spec.system_prompt) == 1

    func = rubric_reward_func(get_rubric(spec.rubric))
    rewards = func(
        [row["prompt"]] * 2,
        [f"the answer is \\boxed{{{row['answer']}}}", "junk"],
        answer=[row["answer"]] * 2,
    )
    assert rewards == [1.0, 0.0]


def _hub_pkg_installed(pkg: str) -> bool:
    if not _venv_ready():
        return False
    probe = subprocess.run(
        [str(hub_python()), "-c", f"import {pkg}"], capture_output=True
    )
    return probe.returncode == 0


@pytest.mark.skipif(
    not _hub_pkg_installed("ifeval"),
    reason="no .venv-prime with ifeval (scripts/setup_prime_venv.sh "
    "primeintellect/ifeval)",
)
def test_ifeval_task_end_to_end():
    from grpo_es.tasks.base import build_dataset, build_eval_dataset
    from grpo_es.tasks.registry import get_task_spec

    spec = get_task_spec("ifeval")
    assert spec.format_scaffold is False
    assert spec.metric_label == "instruction_rate"
    assert spec.eval_size == 100

    # IFEval publishes one 541-row eval-only split; the training pool must
    # exclude exactly the pinned holdout window.
    train = build_dataset(spec, split="train", seed=0)
    held_out = build_eval_dataset(spec)
    assert len(train) == 441 and len(held_out) == 100
    assert set(train["prompt"]).isdisjoint(held_out["prompt"])

    # The reward is the graded rate, not the all-or-nothing pass/fail.
    rubric = get_rubric(spec.rubric)
    row = train[0]
    state = asyncio.run(
        rubric.score_rollout(
            {
                "prompt": row["prompt"],
                "completion": "Way too short, and it even has commas.",
                "answer": row["answer"],
                "info": row["info"],
            }
        )
    )
    assert state["reward"] == state["metrics"]["followed_instructions_rate"]
    assert 0.0 <= state["reward"] < 1.0
    assert state["metrics"]["followed_instructions"] == 0.0
