#!/usr/bin/env python3
"""Live inspector for a GRPO/ES run's rollouts.

Standalone: stdlib only, no project imports, no torch. It tails the
``inspect.jsonl`` a run writes by default (unless ``--no-inspect-dump``) and prints an
append-only, colored log of what the policy is actually producing — one block
per training step. Within a step, each prompt is printed once in full, with all
member responses to that prompt grouped beneath it. Each completion is graded:

    green  = correct answer        (task_reward >= --correct, default 1.0)
    yellow = format-only / partial (decent scaffold or partial reward, wrong)
    red    = wrong

Each line also shows token usage ``tok=used/max`` and flags ``CLIP`` (red) when
the completion ran to the cap without stopping, and shows ``NULL`` (red) in
place of an empty / zero-token response, so degenerate rollouts stand out.

Usage:
    python inspect_run.py                             # auto-follow the newest run under outputs/
    python inspect_run.py outputs/grpo-toy            # tail a specific run dir's inspect.jsonl
    python inspect_run.py --file path/to/inspect.jsonl
    python inspect_run.py --search-root runs          # auto-follow newest under a custom root
    python inspect_run.py outputs/es-toy --no-follow -n 2   # replay last 2 steps, exit

With no run dir given it picks the most recently written ``inspect.jsonl`` it can
find, waits if a run hasn't started yet, and hot-switches to a newer run's dump
when one appears — so you can leave it open and just start training.
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


def _default_search_root() -> Path:
    """Where to hunt for run dumps when none was named: prefer ``outputs/``."""
    out = Path("outputs")
    return out if out.is_dir() else Path(".")


def _find_latest_jsonl(root: Path) -> Path | None:
    """Newest ``inspect.jsonl`` anywhere under ``root`` by mtime, or None."""
    try:
        candidates = [p for p in root.rglob("inspect.jsonl") if p.is_file()]
    except OSError:
        return None
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


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
        self.buf: list[dict] = []  # records of the step currently being buffered
        self.rendered = False  # has self.buf already been printed?

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

    def _render_line(self, rec: dict, tally: dict) -> None:
        grade = self._grade(rec)
        tally[grade] += 1
        clipped = rec.get("clipped", False)
        if clipped:
            tally["clip"] += 1

        sign = rec.get("sign") or ""
        member = f"m{rec.get('member', 0)}{sign}"
        tokens = rec.get("tokens", 0)
        tok = f"{tokens}/{rec.get('max_tokens', 0)}"
        clip_flag = self._c("red", " CLIP") if clipped else ""
        meta = (
            f"ans={rec.get('task_reward', 0.0):.2f} "
            f"fmt={rec.get('format', 0.0):.2f} tok={tok}{clip_flag}"
        )
        snippet = self._clip(
            rec.get("completion", ""), 0 if self.args.full else self.args.width
        )
        if not snippet:
            tally["null"] += 1
            snippet = self._c("red", "NULL")
        dot = self._c(grade, "●")
        print(f"  {dot} {member:<6} {meta}  {self._c('dim', '|')} {snippet}")

    def _flush_step(self) -> None:
        """Render the buffered step: header, each prompt once (full), then all
        member responses to that prompt, then the tally footer.

        Idempotent: a step is flushed either when the tail goes idle (the burst
        for a step is written atomically, so idle == step complete) or when the
        next step's records arrive, whichever lands first; the ``rendered`` guard
        keeps the second trigger from reprinting it."""
        if self.cur_step is None or not self.buf or self.rendered:
            return
        self.rendered = True

        method = self.buf[0].get("method", "?")
        header = f"── step {self.cur_step} · {method} {self.task} "
        print(self._c("bold", header + "─" * max(0, 60 - len(header))))

        # Group records by prompt (group index), preserving first-seen order,
        # so every member's response to a prompt sits under that one prompt.
        groups: dict = {}
        order: list = []
        for rec in self.buf:
            g = rec.get("group", 0)
            if g not in groups:
                groups[g] = []
                order.append(g)
            groups[g].append(rec)

        tally = {"green": 0, "yellow": 0, "red": 0, "clip": 0, "null": 0}
        for g in order:
            recs = groups[g]
            prompt = " ".join(recs[0].get("prompt", "").split())  # full, uncropped
            print(self._c("cyan", f" prompt[{g}]: ") + self._c("dim", prompt))
            for rec in recs:
                self._render_line(rec, tally)

        footer = (
            f"   {self._c('green', '● %d correct' % tally['green'])}  "
            f"{self._c('yellow', '● %d format' % tally['yellow'])}  "
            f"{self._c('red', '● %d wrong' % tally['red'])}  "
            f"{self._c('dim', '%d clipped' % tally['clip'])}  "
            f"{self._c('red', '%d null' % tally['null'])}"
        )
        print(footer)
        print()

    def feed(self, rec: dict) -> None:
        step = rec.get("step")
        if step != self.cur_step:
            self._flush_step()
            self.cur_step = step
            self.buf = []
        self.buf.append(rec)
        self.rendered = False

    def finish(self) -> None:
        self._flush_step()
        self.cur_step = None  # don't double-flush
        self.buf = []


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
    p.add_argument("--search-root", help="auto-follow the newest inspect.jsonl under this dir (default: auto when no run dir given)")
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

    # Auto-discovery mode: with no run dir / --file given (the bare
    # ``inspect_run.py`` case), or with an explicit --search-root, follow the
    # newest inspect.jsonl we can find rather than a fixed path. An explicit run
    # dir or --file is always honored as-is, so a viewer can still be aimed at a
    # run before it has written anything.
    auto = args.search_root is not None or (args.file is None and args.run == ".")
    root = Path(args.search_root) if args.search_root else _default_search_root()

    if auto:
        jsonl = _find_latest_jsonl(root)
        if jsonl is None:
            if not args.follow:
                print(f"no inspect.jsonl found under {root}/", file=sys.stderr)
                return 1
            print(f"waiting for a run to write inspect.jsonl under {root}/ …", file=sys.stderr)
            while jsonl is None:
                time.sleep(0.5)
                jsonl = _find_latest_jsonl(root)
        cfg_path = jsonl.parent / "run_config.json"
        print(f"inspecting {jsonl}", file=sys.stderr)
    else:
        jsonl, cfg_path = _resolve_paths(args)
        # Wait for the file to appear (training may not have dumped yet).
        if not jsonl.exists():
            if not args.follow:
                print(f"no such file: {jsonl}", file=sys.stderr)
                return 1
            print(f"waiting for {jsonl} …", file=sys.stderr)
            while not jsonl.exists():
                time.sleep(0.5)

    renderer = None
    try:
        while True:  # one pass per run; loops again only on a hot-switch (auto)
            # Load run_config.json now — for a viewer started before training,
            # it only exists once the run dir has been written.
            renderer = Renderer(args, _load_cfg(cfg_path))

            # Replay history (optionally only the last N steps), then tail from
            # the current end of file.
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
            switch_to = None
            while switch_to is None:
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
                    # No new bytes: a step is written in one atomic burst, so an
                    # idle read means the buffered step is complete — render it
                    # now instead of waiting for the next step to start.
                    renderer._flush_step()
                    sys.stdout.flush()
                    time.sleep(0.5)
                    # While idle, see if a newer run has started writing.
                    if auto:
                        latest = _find_latest_jsonl(root)
                        if (latest is not None and latest != jsonl
                                and latest.stat().st_mtime > jsonl.stat().st_mtime):
                            switch_to = latest

            # A newer run started — close out the current block and follow it
            # from the top.
            renderer.finish()
            print(f"\n── switched to {switch_to} ", file=sys.stderr)
            jsonl = switch_to
            cfg_path = jsonl.parent / "run_config.json"
            args.last = 0  # show the new run from its start
    except KeyboardInterrupt:
        if renderer is not None:
            renderer.finish()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
