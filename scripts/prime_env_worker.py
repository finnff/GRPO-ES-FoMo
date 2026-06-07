"""Hub-env worker: the only process that runs ``load_environment``.

Launched by ``grpo_es.tasks.from_env.HubEnvClient`` with the ``.venv-prime``
python (hub-env dependencies must never enter the training env — see
``scripts/setup_prime_venv.sh``). Speaks one JSON object per line over
stdin/stdout: requests are ``{"op": ..., ...}``, replies are
``{"ok": true, ...}`` or ``{"ok": false, "error": "<traceback>"}``. A failed
request never kills the worker; only ``op=shutdown`` (or EOF on stdin) does.

Deliberately self-contained: this file runs under a different interpreter
with different site-packages, so it cannot import anything from grpo_es.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import traceback

# Hub envs and HF datasets print progress to stdout, and one stray line would
# corrupt the JSON protocol. Keep a private handle on the real fd 1 for
# replies, then point fd 1 (and sys.stdout) at stderr for everything else.
_PROTO = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)
sys.stdout = sys.stderr

_ENVS: dict[str, object] = {}


def _reply(payload: dict) -> None:
    _PROTO.write(json.dumps(payload, default=str) + "\n")
    _PROTO.flush()


def _get_env(env_id: str) -> object:
    if env_id not in _ENVS:
        raise RuntimeError(f"env {env_id!r} not loaded; send op=load first")
    return _ENVS[env_id]


def _as_messages(value: object, role: str) -> list[dict]:
    """Chat-message list passthrough; bare strings get wrapped."""
    if isinstance(value, list):
        return value
    return [{"role": role, "content": str(value or "")}]


def _op_ping(req: dict) -> dict:
    return {"op": "pong", "python": sys.executable}


def _op_load(req: dict) -> dict:
    # Lazy import on purpose: the protocol tests drive this worker with the
    # training env's python, where verifiers' hub loader doesn't resolve.
    import verifiers
    from verifiers import load_environment

    env_id = req["env_id"]
    if env_id not in _ENVS:
        _ENVS[env_id] = load_environment(env_id, **(req.get("kwargs") or {}))
    env = _ENVS[env_id]
    return {
        "system_prompt": getattr(env, "system_prompt", "") or "",
        "env_class": type(env).__name__,
        "env_mro": [klass.__name__ for klass in type(env).__mro__],
        "verifiers_version": getattr(verifiers, "__version__", "unknown"),
    }


def _op_dataset(req: dict) -> dict:
    env = _get_env(req["env_id"])
    split = req.get("split", "train")

    ds, source = None, "dataset"
    if split != "train" and hasattr(env, "get_eval_dataset"):
        try:
            ds = env.get_eval_dataset()
        except Exception:
            ds = None
        if ds is not None:
            source = "eval_dataset"
    if ds is None:
        ds = env.get_dataset()
    # `source` tells the parent whether a non-train split really got held-out
    # data or silently fell back to the train rows.
    return {"rows": [dict(row) for row in ds], "source": source}


async def _score_one(rubric: object, item: dict) -> tuple[float, dict]:
    prompt = _as_messages(item.get("prompt"), "user")
    completion = _as_messages(item.get("completion"), "assistant")
    state = {
        "prompt": prompt,
        "completion": completion,
        "answer": item.get("answer", ""),
        "info": item.get("info") or {},
        "task": item.get("task") or "default",
        "example_id": 0,
        # Synthesized single-turn trajectory: metric funcs inherited from
        # MultiTurnEnv (num_turns, ...) read state["trajectory"] and error
        # per rollout without it.
        "trajectory": [
            {
                "prompt": prompt,
                "completion": completion,
                "response": None,
                "tokens": None,
                "reward": None,
                "advantage": None,
                "is_truncated": False,
                "trajectory_id": "local",
                "extras": {},
            }
        ],
    }

    # Bridge both rubric generations: 0.1.x takes one state dict and mutates
    # it; newer verifiers take keyword args and return a result object.
    # Detection keys off the "prompt" parameter — brittle if a future rubric
    # renames it, but there's no version handshake to lean on instead.
    fn = rubric.score_rollout
    params = inspect.signature(fn).parameters
    has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
    if "prompt" in params or has_var_kw:
        out = fn(**state, state=state)
    else:
        out = fn(state)
    if inspect.isawaitable(out):
        out = await out

    reward = getattr(out, "reward", None)
    metrics = getattr(out, "metrics", None)
    if reward is None and isinstance(out, dict):
        reward, metrics = out.get("reward"), out.get("metrics")
    if reward is None:  # 0.1.x mutated the state instead of returning
        reward, metrics = state.get("reward"), state.get("metrics")
    return float(reward or 0.0), dict(metrics or {})


def _op_score(req: dict) -> dict:
    env = _get_env(req["env_id"])
    rubric = env.rubric

    async def run() -> list[tuple[float, dict]]:
        return await asyncio.gather(
            *(_score_one(rubric, item) for item in req["rollouts"])
        )

    scored = asyncio.run(run())
    return {
        "rewards": [reward for reward, _ in scored],
        "metrics": [metrics for _, metrics in scored],
    }


_HANDLERS = {
    "ping": _op_ping,
    "load": _op_load,
    "dataset": _op_dataset,
    "score": _op_score,
}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "shutdown":
                _reply({"ok": True})
                return
            handler = _HANDLERS.get(op)
            if handler is None:
                _reply({"ok": False, "error": f"unknown op {op!r}"})
                continue
            out = handler(req)
            out["ok"] = True
            _reply(out)
        except Exception:
            _reply({"ok": False, "error": traceback.format_exc()})


if __name__ == "__main__":
    main()
