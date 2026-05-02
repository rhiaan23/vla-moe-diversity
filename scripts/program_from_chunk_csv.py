"""Build a manual-routing program JSON from a recorded chunk_routing.csv.

Usage:
  python scripts/program_from_chunk_csv.py \
    outputs/evals/v5_libero_10/chunk_routing.csv \
    --task=6 --episode=0 \
    --out=programs/t6_router_replay_ep0.json

The output JSON has the schema consumed by ``eval_v5_manual_routing.py``:
  {"by_chunk": [{"indices": [...], "weights": [...]}, ...], "default": {...}}

The default is set to the last chunk's choice so episodes that run for more
chunks than the source episode keep doing something sensible.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows: list[dict] = []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            if int(row["task"]) == args.task and int(row["episode"]) == args.episode:
                rows.append(row)
    rows.sort(key=lambda r: int(r["chunk"]))
    if not rows:
        raise SystemExit(f"no rows for task={args.task} episode={args.episode}")

    by_chunk = []
    for row in rows:
        idx = [int(row["top1_expert"]), int(row["top2_expert"])]
        wts = [float(row["top1_weight"]), float(row["top2_weight"])]
        by_chunk.append({"indices": idx, "weights": wts})

    program = {
        "source_csv": str(args.csv),
        "source_task": args.task,
        "source_episode": args.episode,
        "n_chunks": len(by_chunk),
        "by_chunk": by_chunk,
        "default": by_chunk[-1],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(program, f, indent=2)
    print(f"wrote {args.out} ({len(by_chunk)} chunks; last={by_chunk[-1]})")


if __name__ == "__main__":
    main()
