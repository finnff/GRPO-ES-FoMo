"""Hub-env adapter: worker protocol, prompt flattening, dynamic registries.

Everything except the last test runs under the normal training env — the
worker imports verifiers lazily, so the protocol tests can drive it with
``sys.executable``. The end-to-end test needs a real ``.venv-prime`` with
``primeintellect/gsm8k`` installed and skips cleanly without one.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from grpo_es.config.run_config import task_arg
from grpo_es.rewards.registry import DYNAMIC_RUBRICS, get_rubric, register_rubric
from grpo_es.tasks.from_env import (
    default_prompt_transform,
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
