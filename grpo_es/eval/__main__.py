"""Held-out eval: score the base model and any adapters on a task's eval slice.

    python -m grpo_es.eval --task gsm8k --model qwen3.5-0.8b \
        --adapter base outputs/grpo-gsm8k/checkpoint-final --json out.json

    python -m grpo_es.eval --task mmlu_pro --model qwen3.5-0.8b \
        --adapter outputs/grpo-mmlu/checkpoint-final --kl \
        --decode-from outputs/grpo-mmlu

The table prints (and writes to ``--json``) means; turning per-sample reward
vectors into CIs and p-values is ``grpo_es.eval.stats`` territory, fed by the
``per_sample_reward`` arrays in the JSON payload.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grpo_es.config.logging_setup import configure_logging
from grpo_es.config.run_config import KNOWN_TASKS, task_arg
from grpo_es.eval.metrics import coherence_gate
from grpo_es.eval.runner import DecodeParams, evaluate_adapter, load_tokenizer
from grpo_es.models import resolve_model_alias
from grpo_es.rewards.registry import get_rubric
from grpo_es.tasks.base import apply_chat_template, build_eval_dataset
from grpo_es.tasks.registry import get_task_spec


def _parse_slice(text: str) -> tuple[int, int]:
    """'100:300' -> (offset=100, size=200)."""
    try:
        start, stop = (int(part) for part in text.split(":"))
    except ValueError:
        raise SystemExit(f"--slice wants START:STOP, got {text!r}") from None
    if stop <= start:
        raise SystemExit(f"--slice STOP must exceed START, got {text!r}")
    return start, stop - start


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m grpo_es.eval",
        description="Score base model + adapters on a task's held-out slice.",
    )
    p.add_argument(
        "--task",
        required=True,
        type=task_arg,
        help=f"one of {sorted(KNOWN_TASKS)} or env:<owner>/<env>",
    )
    p.add_argument(
        "--model", required=True, help="HF repo id, local path, or short alias"
    )
    p.add_argument(
        "--adapter",
        nargs="*",
        default=["base"],
        help="adapter dirs to score; 'base' = bare model (always scored first)",
    )
    p.add_argument("--split", default=None, help="override the spec's eval split")
    p.add_argument(
        "--slice",
        default=None,
        metavar="START:STOP",
        help="override the spec's eval slice, e.g. 100:300",
    )
    p.add_argument("--decode", choices=("greedy", "sample"), default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--rep-penalty", type=float, default=None)
    p.add_argument(
        "--decode-from",
        default=None,
        metavar="RUNDIR",
        help="take decode params from RUNDIR/run_config.json (training regime); "
        "explicit decode flags still win per-field",
    )
    p.add_argument("--seed", type=int, default=0, help="generation seed (per adapter)")
    p.add_argument(
        "--chat-template",
        choices=("auto", "on", "off"),
        default="auto",
        help="wrap prompts in the chat template; MUST match how the adapter was "
        "trained (auto = on iff the tokenizer ships one)",
    )
    p.add_argument(
        "--kl",
        action="store_true",
        help="also estimate KL(adapter‖base) over the eval completions",
    )
    p.add_argument("--json", default=None, metavar="OUT.json")
    p.add_argument(
        "--per-sample",
        default=None,
        metavar="OUT.jsonl",
        help="per-prompt records (reward, gate, length, snippet)",
    )
    p.add_argument(
        "--show", type=int, default=0, help="print N sample completions (clipped first)"
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _resolve_decode(args, spec) -> DecodeParams:
    base = (
        DecodeParams.from_run_config(args.decode_from)
        if args.decode_from
        else DecodeParams()
    )
    decode = base.merged(
        decode=args.decode,
        temperature=args.temperature,
        repetition_penalty=args.rep_penalty,
    )
    if decode.max_new is None:
        decode.max_new = spec.eval_max_new
    if decode.max_prompt is None:
        decode.max_prompt = spec.eval_max_prompt
    return decode


def _write_json(
    path: str, spec, model_id: str, decode: DecodeParams, seed: int, results: list, base
) -> None:
    """The means table plus the raw per-sample reward vectors paired stats need."""
    payload = {
        "task": spec.name,
        "model": model_id,
        "metric_label": spec.metric_label,
        "decode": {
            "decode": decode.decode,
            "temperature": decode.temperature,
            "repetition_penalty": decode.repetition_penalty,
            "max_new": decode.max_new,
            "max_prompt": decode.max_prompt,
        },
        "holdout": base.metrics.n,
        "seed": seed,
        "adapters": [
            {
                "adapter": r.label,
                "mean_reward": r.metrics.mean_reward,
                "accuracy": r.metrics.accuracy,
                "format_pass": r.metrics.format_pass,
                "mean_length": r.metrics.mean_length,
                "mean_gate": r.mean_gate,
                "clip_frac": r.clip_frac,
                "kl_to_base": r.kl_to_base,
                "tokens": r.tokens,
                "n": r.metrics.n,
                "delta_mean_reward": r.metrics.mean_reward - base.metrics.mean_reward,
                "delta_accuracy": r.metrics.accuracy - base.metrics.accuracy,
                # The raw vector — paired stats need it, means don't carry it.
                "per_sample_reward": [s["reward"] for s in r.metrics.per_sample],
            }
            for r in results
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


def _write_per_sample(path: str, results: list) -> None:
    """One JSONL record per (adapter, prompt): reward, gate, length, snippet."""
    with open(path, "w") as fh:
        for r in results:
            for i, sample in enumerate(r.metrics.per_sample):
                fh.write(
                    json.dumps(
                        {
                            "adapter": r.label,
                            "i": i,
                            "reward": sample["reward"],
                            "gate": coherence_gate(r.completions[i]),
                            "len": sample["length"],
                            "clipped": r.clipped[i],
                            "snippet": r.completions[i][:200],
                        }
                    )
                    + "\n"
                )
    print(f"wrote {path}")


def _show_samples(result, n: int) -> None:
    # Clipped completions first — they're where the trouble usually is.
    order = sorted(range(len(result.completions)), key=lambda i: not result.clipped[i])
    for i in order[:n]:
        flag = " [clipped]" if result.clipped[i] else ""
        print(f"\n--- {result.label} sample {i}{flag} ---")
        print(result.completions[i][:1500])


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    configure_logging(verbose=args.verbose)

    spec = get_task_spec(args.task)
    model_id = resolve_model_alias(args.model)
    offset, size = _parse_slice(args.slice) if args.slice else (None, None)
    dataset = build_eval_dataset(spec, split=args.split, offset=offset, size=size)
    rubric = get_rubric(spec.rubric)
    decode = _resolve_decode(args, spec)

    # Base first: every delta below is relative to it.
    adapters = ["base"] + [a for a in args.adapter if a != "base"]

    tok = load_tokenizer(model_id)
    dataset = apply_chat_template(dataset, tok, args.chat_template)
    print(
        f"task={spec.name} model={model_id} holdout={len(dataset)} "
        f"decode={decode.label()} chat_template={args.chat_template} max_new={decode.max_new}"
    )
    label = spec.metric_label
    print(
        f"{'adapter':<44} {label:>10} {'exact':>6} {'fmt':>5} "
        f"{'mean_len':>8} {'gate':>5} {'clip%':>6} {'kl':>7}"
    )

    results = []
    for adapter in adapters:
        result = evaluate_adapter(
            model_id,
            adapter,
            tok,
            dataset,
            rubric,
            decode,
            seed=args.seed,
            want_kl=args.kl,
        )
        results.append(result)
        m = result.metrics
        shown = adapter if len(adapter) <= 44 else "..." + adapter[-41:]
        kl_cell = f"{result.kl_to_base:7.4f}" if result.kl_to_base is not None else "      —"
        print(
            f"{shown:<44} {m.mean_reward:>10.4f} {m.accuracy:>6.3f} "
            f"{m.format_pass:>5.2f} {m.mean_length:>8.0f} {result.mean_gate:>5.2f} "
            f"{result.clip_frac:>5.0%} {kl_cell}",
            flush=True,
        )
        if args.show:
            _show_samples(result, args.show)

    base = results[0]
    for result in results[1:]:
        print(
            f"Δ vs base [{result.label}]: "
            f"{label} {result.metrics.mean_reward - base.metrics.mean_reward:+.4f}, "
            f"exact {result.metrics.accuracy - base.metrics.accuracy:+.3f}"
        )

    if args.json:
        _write_json(args.json, spec, model_id, decode, args.seed, results, base)
    if args.per_sample:
        _write_per_sample(args.per_sample, results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
