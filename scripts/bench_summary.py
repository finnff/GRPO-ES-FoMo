#!/usr/bin/env python3
"""Summarize a benchmark dir: step time / VRAM / reward+length trajectory per cell."""
import json, glob, os, sys

out = sys.argv[1] if len(sys.argv) > 1 else "outputs/bench-ifeval"


def traj(cell):
    """Return (reward_first, reward_last, reward_max, len_first, len_last) or Nones."""
    rc = os.path.join(cell, "run_config.json")
    method = json.load(open(rc)).get("method") if os.path.exists(rc) else "?"
    if method == "es":
        h = os.path.join(cell, "history.jsonl")
        if not os.path.exists(h):
            return method, (None,) * 5
        recs = [json.loads(l) for l in open(h) if l.strip()]
        r = [x.get("fitness_best") for x in recs if x.get("fitness_best") is not None]
        L = [x.get("mean_len") for x in recs if x.get("mean_len") is not None]
    else:
        states = glob.glob(os.path.join(cell, "checkpoint-*", "trainer_state.json"))
        best = None
        for s in states:
            d = json.load(open(s))
            if best is None or d.get("global_step", 0) > best.get("global_step", 0):
                best = d
        if not best:
            return method, (None,) * 5
        lh = best["log_history"]
        r = [e["reward"] for e in lh if "reward" in e]
        L = ([e["completions/mean_length"] for e in lh if "completions/mean_length" in e]
             or [e["completion_length"] for e in lh if "completion_length" in e])
    return method, (
        (round(r[0], 3), round(r[-1], 3), round(max(r), 3)) if r else (None, None, None)
    ) + ((round(L[0]), round(L[-1])) if L else (None, None))


rows = []
for cell in sorted(glob.glob(os.path.join(out, "*", ""))):
    name = os.path.basename(cell.rstrip("/"))
    tb = os.path.join(cell, "token_budget.json")
    if not os.path.exists(tb):
        rows.append((name, "NO-BUDGET")); continue
    b = json.load(open(tb))
    rc = json.load(open(os.path.join(cell, "run_config.json")))
    method, (r0, r1, rmax, l0, l1) = traj(cell)
    rows.append((
        name, method, (rc.get("model") or "").split("/")[-1],
        b.get("global_step"), round(b.get("mean_step_time") or 0, 3),
        round(b.get("train_runtime") or 0, 1), round(b.get("tokens_per_second") or 0),
        round((b.get("peak_vram_bytes") or 0) / 1e9, 2),
        f"{r0}->{r1} (max {rmax})", f"{l0}->{l1}",
    ))

hdr = f'{"cell":18}{"meth":5}{"model":22}{"stp":>4}{"s/step":>8}{"run_s":>7}{"tok/s":>7}{"vram":>6}  {"reward":24}{"len":12}'
print("\n" + hdr); print("-" * len(hdr))
for row in rows:
    if len(row) == 2:
        print(f"{row[0]:18}{row[1]}"); continue
    n, m, mod, st, ss, rs, ts, v, rew, ln = row
    print(f'{n:18}{m or "":5}{(mod or "")[:21]:22}{str(st):>4}{ss:>8}{rs:>7}{ts:>7}{v:>6}  {rew:24}{ln:12}')
