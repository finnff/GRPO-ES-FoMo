#!/usr/bin/env python3
"""Live inspector for a GRPO/ES run's rollouts.

Standalone: stdlib only, no project imports, no torch. It tails the
``inspect.jsonl`` a run writes when started with ``--inspect-dump`` and prints an
append-only, colored log of what the policy is actually producing — one block
per training step, each completion graded:

    green  = correct answer        (task_reward >= --correct, default 1.0)
    yellow = format-only / partial (decent scaffold or partial reward, wrong)
    red    = wrong

Each line also shows token usage ``tok=used/max`` and flags ``CLIP`` when the
completion ran to the cap without stopping, so truncated answers stand out.

Usage:
    python inspect_run.py outputs/grpo-toy            # tail the run dir's inspect.jsonl
    python inspect_run.py --file path/to/inspect.jsonl
    python inspect_run.py outputs/es-toy --no-follow -n 2   # replay last 2 steps, exit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ESC = "\033["
COLORS = {
    "green": f"{ESC}32m",
    "yellow": f"{ESC}33m",
    "red": f"{ESC}31m",
    "dim": f"{ESC}2m",
    "bold": f"{ESC}1m",
    "cyan": f"{ESC}36m",
    "reset": f"{ESC}0m",
}


def _resolve_paths(args) -> tuple[Path, Path]:
    """Return (jsonl_path, run_config_path)."""
    if args.file:
        jsonl = Path(args.file)
        cfg_dir = jsonl.parent
    else:
        target = Path(args.run)
        # A path that already names a file, or ends in .jsonl, is the dump
        # itself; anything else is a run dir (which may not exist yet — the
        # viewer can be started before training has written anything).
        if target.suffix == ".jsonl" or (target.exists() and target.is_file()):
            jsonl = target
            cfg_dir = target.parent
        else:
            jsonl = target / "inspect.jsonl"
            cfg_dir = target
    return jsonl, cfg_dir / "run_config.json"


def _load_cfg(cfg_path: Path) -> dict:
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


class Renderer:
    """Stateful formatter: feed records in file order, it prints step blocks.

    A step's footer (the green/yellow/red tally) is flushed when the next step
    arrives — natural for an append-only stream where a step's end is only known
    once the following one starts.
    """

    def __init__(self, args, cfg: dict) -> None:
        self.args = args
        self.task = cfg.get("task", "?")
        self.color = args.color
        self.cur_step = None
        self.cur_group = None
        self.tally = {"green": 0, "yellow": 0, "red": 0, "clip": 0}

    def _c(self, name: str, text: str) -> str:
        if not self.color:
            return text
        return f"{COLORS[name]}{text}{COLORS['reset']}"

    def _grade(self, rec: dict) -> str:
        tr = rec.get("task_reward", 0.0)
        fmt = rec.get("format", 0.0)
        if tr >= self.args.correct:
            return "green"
        if fmt >= 0.5 or tr > 0:
            return "yellow"
        return "red"

    @staticmethod
    def _clip(text: str, width: int) -> str:
        flat = " ".join(text.split())
        if width and len(flat) > width:
            return flat[:width] + "…"
        return flat

    def _flush_footer(self) -> None:
        if self.cur_step is None:
            return
        t = self.tally
        line = (
            f"   {self._c('green', '● %d correct' % t['green'])}  "
            f"{self._c('yellow', '● %d format' % t['yellow'])}  "
            f"{self._c('red', '● %d wrong' % t['red'])}  "
            f"{self._c('dim', '%d clipped' % t['clip'])}"
        )
        print(line)
        print()

    def feed(self, rec: dict) -> None:
        step = rec.get("step")
        method = rec.get("method", "?")
        if step != self.cur_step:
            self._flush_footer()
            self.tally = {"green": 0, "yellow": 0, "red": 0, "clip": 0}
            self.cur_step = step
            self.cur_group = None
            header = f"── step {step} · {method} {self.task} "
            print(self._c("bold", header + "─" * max(0, 60 - len(header))))

        group = rec.get("group", 0)
        if group != self.cur_group:
            self.cur_group = group
            prompt = self._clip(
                rec.get("prompt", ""), 0 if self.args.full else 160
            )
            print(self._c("cyan", f" prompt[{group}]: ") + self._c("dim", prompt))

        grade = self._grade(rec)
        self.tally[grade] += 1
        clipped = rec.get("clipped", False)
        if clipped:
            self.tally["clip"] += 1

        sign = rec.get("sign") or ""
        member = f"m{rec.get('member', 0)}{sign}"
        tok = f"{rec.get('tokens', 0)}/{rec.get('max_tokens', 0)}"
        clip_flag = self._c("red", " CLIP") if clipped else ""
        meta = (
            f"ans={rec.get('task_reward', 0.0):.2f} "
            f"fmt={rec.get('format', 0.0):.2f} tok={tok}{clip_flag}"
        )
        snippet = self._clip(
            rec.get("completion", ""), 0 if self.args.full else self.args.width
        )
        dot = self._c(grade, "●")
        print(f"  {dot} {member:<6} {meta}  {self._c('dim', '|')} {snippet}")

    def finish(self) -> None:
        self._flush_footer()
        self.cur_step = None  # don't double-flush


def _read_records(path: Path) -> list[dict]:
    records = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _last_n_steps(records: list[dict], n: int) -> list[dict]:
    steps = []
    for r in records:
        s = r.get("step")
        if s not in steps:
            steps.append(s)
    keep = set(steps[-n:]) if n > 0 else set(steps)
    return [r for r in records if r.get("step") in keep]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", default=".", help="run output dir (reads inspect.jsonl)")
    p.add_argument("--file", help="path to an inspect.jsonl directly")
    p.add_argument("-f", "--follow", dest="follow", action="store_true", default=True, help="tail the file (default)")
    p.add_argument("--no-follow", dest="follow", action="store_false", help="render and exit")
    p.add_argument("-n", "--last", type=int, default=0, metavar="N", help="render only the last N steps already on disk first")
    p.add_argument("--width", type=int, default=200, help="truncate completions to this many chars (default 200)")
    p.add_argument("--full", action="store_true", help="never truncate prompts/completions")
    p.add_argument("--correct", type=float, default=1.0, help="green threshold on task_reward (default 1.0)")
    p.add_argument("--no-color", dest="color", action="store_false", default=None, help="disable ANSI colors")
    args = p.parse_args(argv)

    if args.color is None:
        args.color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    jsonl, cfg_path = _resolve_paths(args)

    # Wait for the file to appear (training may not have dumped yet).
    if not jsonl.exists():
        if not args.follow:
            print(f"no such file: {jsonl}", file=sys.stderr)
            return 1
        print(f"waiting for {jsonl} …", file=sys.stderr)
        while not jsonl.exists():
            time.sleep(0.5)

    # Load run_config.json now — for a viewer started before training, it only
    # exists once the run dir has been written.
    renderer = Renderer(args, _load_cfg(cfg_path))

    # Replay history (optionally only the last N steps), then start tailing
    # from the current end of file.
    existing = _read_records(jsonl)
    if args.last > 0:
        existing = _last_n_steps(existing, args.last)
    for rec in existing:
        renderer.feed(rec)

    if not args.follow:
        renderer.finish()
        return 0

    pos = jsonl.stat().st_size
    buf = ""
    try:
        while True:
            with jsonl.open() as fh:
                fh.seek(pos)
                chunk = fh.read()
                pos = fh.tell()
            if chunk:
                buf += chunk
                *lines, buf = buf.split("\n")
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        renderer.feed(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                sys.stdout.flush()
            else:
                time.sleep(0.5)
    except KeyboardInterrupt:
        renderer.finish()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
