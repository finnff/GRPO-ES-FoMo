"""PrimeIntellect Environments Hub adapter: ``--task env:<owner>/<env>``.

A single-turn hub environment is a dataset + a rubric behind
``load_environment(env_id)`` — exactly the two pieces a TaskSpec needs. But
hub envs are pip wheels whose dependency trees (fresh verifiers, newer
openai) don't co-resolve with this repo's pins, so ``load_environment`` runs
in a dedicated venv (``scripts/setup_prime_venv.sh``) behind a worker
subprocess speaking JSON lines (``scripts/prime_env_worker.py``). This module
is the parent side: the worker client, a Rubric facade that scores remotely,
and ``task_from_environment`` which wires both into the normal registries.

Hard boundary: only the env's dataset + rubric are consumed. Multi-turn,
tool, and sandbox envs need the env's own rollout harness, which both
optimizer legs bypass — they generate locally.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

from datasets import Dataset
from verifiers.rubrics.rubric import Rubric

from grpo_es.rewards.registry import register_rubric
from grpo_es.tasks.base import TaskSpec, shuffle_take
from grpo_es.tasks.registry import register_task

logger = logging.getLogger(__name__)

ENV_TASK_PREFIX = "env:"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER = _REPO_ROOT / "scripts" / "prime_env_worker.py"
_DEFAULT_VENV_PYTHON = _REPO_ROOT / ".venv-prime" / "bin" / "python"


def hub_python() -> Path:
    """The isolated venv's interpreter ($PRIME_ENV_PYTHON overrides)."""
    override = os.environ.get("PRIME_ENV_PYTHON")
    python = Path(override) if override else _DEFAULT_VENV_PYTHON
    if not python.exists():
        raise FileNotFoundError(
            f"hub-env venv python not found at {python}. Build the isolated "
            f"venv first:  scripts/setup_prime_venv.sh <owner>/<env> ...  "
            f"(or point $PRIME_ENV_PYTHON at an existing venv's python). "
            f"Hub-env deps must NOT be installed into the training env."
        )
    return python


class HubEnvClient:
    """One persistent worker subprocess, JSON-line RPC over stdin/stdout.

    A dead worker is restarted on the next request, but its loaded envs die
    with it — ``task_from_environment`` re-issues ``load`` per task, which
    covers the restart case for everything registered up front.
    """

    _shared: HubEnvClient | None = None

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None

    @classmethod
    def shared(cls) -> HubEnvClient:
        if cls._shared is None:
            cls._shared = cls()
            atexit.register(cls._shared.close)
        return cls._shared

    def _ensure(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            python = hub_python()
            logger.info("starting hub-env worker: %s %s", python, _WORKER)
            # stderr stays inherited so env install / dataset download
            # progress remains visible.
            self._proc = subprocess.Popen(
                [str(python), str(_WORKER)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
        return self._proc

    def request(self, **req: Any) -> dict:
        proc = self._ensure()
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()  # no timeout: rubrics own their limits
        if not line:
            code = proc.poll()
            self._proc = None
            raise RuntimeError(
                f"hub-env worker died (exit={code}) on op={req.get('op')!r}; "
                f"check its stderr above"
            )
        resp = json.loads(line)
        if not resp.get("ok"):
            raise RuntimeError(
                f"hub-env worker op={req.get('op')!r} failed:\n{resp.get('error')}"
            )
        return resp

    def load(self, env_id: str, **kwargs: Any) -> dict:
        return self.request(op="load", env_id=env_id, kwargs=kwargs)

    def dataset(self, env_id: str, split: str) -> tuple[list[dict], str]:
        resp = self.request(op="dataset", env_id=env_id, split=split)
        return resp["rows"], resp["source"]

    def score(
        self, env_id: str, rollouts: list[dict]
    ) -> tuple[list[float], list[dict]]:
        resp = self.request(op="score", env_id=env_id, rollouts=rollouts)
        return resp["rewards"], resp["metrics"]

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self.request(op="shutdown")
            except Exception:
                self._proc.kill()
        self._proc = None


def messages_text(value: Any) -> str:
    """Collapse a chat-message list back to text; strings pass through."""
    if isinstance(value, list):
        return "\n\n".join(str(m.get("content", "")) for m in value)
    return str(value or "")


def default_prompt_transform(prompt: Any, system_prompt: str) -> str:
    """Flatten an env's chat prompt to the spine's raw-string format.

    No chat template — same reproducibility invariant as eval generation.
    Envs that already bake their system prompt into the chat messages
    (gsm8k does) must not get it prepended a second time.
    """
    body = messages_text(prompt)
    has_system = isinstance(prompt, list) and any(
        m.get("role") == "system" for m in prompt
    )
    if system_prompt and not has_system:
        return f"{system_prompt}\n\n{body}"
    return body


class RemoteEnvRubric(Rubric):
    """A verifiers Rubric whose scoring happens in the hub-env worker.

    Satisfies the same async ``score_rollout(state)`` contract the local
    rubrics speak, so the TRL bridge and the eval runner need no changes.
    Each rollout is one worker round-trip; the wire protocol can batch, but
    the per-rollout Rubric interface is the seam everything else calls.
    """

    def __init__(self, env_id: str) -> None:
        super().__init__()
        self.env_id = env_id

    async def score_rollout(self, state: dict, **_: Any) -> dict:
        item = {
            "prompt": messages_text(state.get("prompt")),
            "completion": messages_text(state.get("completion")),
            "answer": state.get("answer", ""),
            "info": state.get("info") or {},
            "task": state.get("task") or "default",
        }
        rewards, metrics = HubEnvClient.shared().score(self.env_id, [item])
        state["reward"] = rewards[0]
        state["metrics"] = metrics[0]
        return state


def task_from_environment(
    name: str,
    env_id: str,
    *,
    rubric_override: str | None = None,
    prompt_transform: Callable[[Any, str], str] | None = None,
    eval_split: str = "test",
    eval_seed: int = 0,
    eval_offset: int = 0,
    eval_size: int | None = None,
    eval_max_prompt: int = 512,
    eval_max_new: int = 768,
    metric_label: str = "score",
    register: bool = True,
    **load_kwargs: Any,
) -> tuple[TaskSpec, Callable[..., Dataset]]:
    """Build (and by default register) a TaskSpec + loader from a hub env.

    ``rubric_override`` keeps a spine rubric where the env's stock rubric is
    wrong for the gradient (all-or-nothing scoring collapses within-group
    advantage); ``eval_offset``/``eval_size`` carve a disjoint holdout when
    the env publishes only one split.
    """
    client = HubEnvClient.shared()
    meta = client.load(env_id, **load_kwargs)
    if "SingleTurnEnv" not in meta.get("env_mro", []):
        logger.warning(
            "env %s is %s, not a SingleTurnEnv. Only its dataset+rubric are "
            "used; multi-turn/tool behavior needs the env's rollout harness, "
            "which both optimizer legs bypass — rollouts are scored as "
            "single-turn.",
            env_id,
            meta.get("env_class"),
        )

    system_prompt = meta.get("system_prompt", "")
    transform = prompt_transform or default_prompt_transform

    def build_prompt(row: dict) -> str:
        return transform(row.get("prompt"), system_prompt)

    spec = TaskSpec(
        name=name,
        rubric=rubric_override or name,
        system_prompt=system_prompt,
        build_prompt=build_prompt,
        format_scaffold=False,  # env rubrics grade the raw response
        eval_split=eval_split,
        eval_seed=eval_seed,
        eval_offset=eval_offset,
        eval_size=eval_size,
        eval_max_prompt=eval_max_prompt,
        eval_max_new=eval_max_new,
        metric_label=metric_label,
    )

    def load_env_task(
        spec: TaskSpec,
        split: str = "train",
        seed: int = 0,
        max_samples: int | None = None,
    ) -> Dataset:
        rows, source = client.dataset(env_id, split)
        if split != "train" and source != "eval_dataset":
            logger.warning(
                "env %s has no eval split; split=%r fell back to its train "
                "dataset — the held-out slice is NOT disjoint from training. "
                "Carve a disjoint window with eval_offset/eval_size.",
                env_id,
                split,
            )
        has_info = any(row.get("info") for row in rows)
        mapped = []
        for row in rows:
            item = {
                "prompt": spec.build_prompt(row),
                "answer": "" if row.get("answer") is None else str(row["answer"]),
            }
            if has_info:
                item["info"] = row.get("info") or {}
            if row.get("task") is not None:
                item["task"] = row["task"]
            mapped.append(item)
        return shuffle_take(Dataset.from_list(mapped), seed, max_samples)

    if register:
        register_task(spec, load_env_task)
        if rubric_override is None:
            register_rubric(spec.rubric, lambda: RemoteEnvRubric(env_id))
    return spec, load_env_task


def register_environment_task(task_name: str) -> TaskSpec:
    """CLI entry point: lazily register ``env:<owner>/<env>`` on first use."""
    if not task_name.startswith(ENV_TASK_PREFIX):
        raise ValueError(f"expected 'env:<owner>/<env>', got {task_name!r}")
    env_id = task_name[len(ENV_TASK_PREFIX) :]
    if not env_id:
        raise ValueError(f"empty env id in task name {task_name!r}")
    spec, _ = task_from_environment(task_name, env_id)
    return spec
