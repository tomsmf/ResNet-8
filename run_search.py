"""Run grid or random search over several seeds, resumable.

One JSON line per evaluation in results/<method>_seed<seed>.jsonl, written as soon
as the evaluation finishes. Restarting the same command skips finished evaluations.

Example:
    caffeinate -i python run_search.py --method grid
    caffeinate -i python run_search.py --method random
"""

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

import data
from objective import get_device, run_evaluation
from search_space import grid_configs, random_configs

SEEDS = [42, 43, 44, 45, 46]


def load_done(path: Path) -> dict[int, dict]:
    """Read finished evaluations; drop a half-written last line from a crash."""
    if not path.exists():
        return {}
    done, good_lines = {}, []
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        done[rec["eval_idx"]] = rec
        good_lines.append(line)
    path.write_text("".join(l + "\n" for l in good_lines))
    return done


def append_record(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())  # make sure it's on disk even if the machine dies right after


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["grid", "random"], required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--budget", type=int, default=100)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()

    device = get_device()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"device": str(device), "host": platform.node(), "torch": torch.__version__}
    print(f"method={args.method} seeds={args.seeds} device={device}", flush=True)

    for seed in args.seeds:
        if args.method == "grid":
            configs = grid_configs()
            assert len(configs) <= args.budget, f"grid ({len(configs)}) exceeds budget ({args.budget})"
        else:
            configs = random_configs(args.budget, seed)

        path = args.out / f"{args.method}_seed{seed}.jsonl"
        done = load_done(path)
        for idx, rec in done.items():
            # Guard against resuming with a changed search space.
            assert rec["config"] == configs[idx], f"{path}: config {idx} differs from current search space"
        todo = [i for i in range(len(configs)) if i not in done]
        print(f"seed {seed}: {len(done)} done, {len(todo)} to go -> {path}", flush=True)
        if not todo:
            continue

        split = data.load_split(seed, device)
        for n, idx in enumerate(todo, 1):
            rec = run_evaluation(configs[idx], seed, args.epochs, split=split, device=device)
            rec = {"method": args.method, "eval_idx": idx, **rec, **meta,
                   "finished_at": datetime.now(timezone.utc).isoformat()}
            append_record(path, rec)
            print(f"[{args.method} s{seed} {idx + 1:3d}/{len(configs)}] score={rec['score']:.4f} "
                  f"{'DIVERGED ' if rec['diverged'] else ''}{rec['wall_time_s']:.0f}s  {rec['config']}",
                  flush=True)


if __name__ == "__main__":
    main()
