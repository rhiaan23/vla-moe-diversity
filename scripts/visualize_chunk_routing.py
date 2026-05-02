"""Visualize per-chunk expert dispatch for one or more episodes.

Produces a strip plot: rows = (condition, episode), x-axis = chunk_idx,
cell = top-1 expert id (color-coded). Useful for showing how the
router (or a manual program) sequences experts within a trajectory.

Usage:
  python scripts/visualize_chunk_routing.py \\
    --routing_logs \\
        outputs/evals/v5_libero_10/chunk_routing.csv:router_T5 \\
        outputs/evals/manual_routing/family1_essential/replay_t5_ep0_on_t5/routing_log.json:replay_T5 \\
        outputs/evals/manual_routing/family1_essential/phase_t5_approach_then_transport/routing_log.json:phase_T5 \\
    --tasks 5 --out figures/chunk_routing_T5.png
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _load_csv(path: Path, want_task: int | None) -> list[tuple[int, int]]:
    """Return list of (chunk, top1_expert) per (task, episode); flatten."""
    out = []
    last_te = None
    seq: list[int] = []
    rows = sorted(csv.DictReader(open(path)), key=lambda r: (int(r["task"]), int(r["episode"]), int(r["chunk"])))
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row in rows:
        t, e = int(row["task"]), int(row["episode"])
        if want_task is not None and t != want_task:
            continue
        grouped[(t, e)].append(int(row["top1_expert"]))
    return [(label, seq) for label, seq in sorted(grouped.items())]


def _load_json(path: Path, want_task: int | None) -> list[tuple[tuple[int, int], list[int]]]:
    """For manual_routing routing_log.json: returns [((task, ep_batch), [top1_per_chunk])]."""
    log = json.load(open(path))
    grouped: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for r in log:
        t = r["task"]
        if want_task is not None and t != want_task:
            continue
        ep = r.get("episode_batch", 0)
        grouped[(t, ep)].append((r["chunk_idx"], r["manual_indices"][0][0]))  # env 0, top-1
    out = []
    for k, lst in sorted(grouped.items()):
        lst.sort(key=lambda x: x[0])
        out.append((k, [e for _, e in lst]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing_logs", nargs="+", required=True,
                    help="path:label entries; path is .csv (router) or .json (manual)")
    ap.add_argument("--tasks", default="", help="comma-separated tasks to filter; empty = all")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max_episodes_per_label", type=int, default=5)
    args = ap.parse_args()

    want_tasks = [int(t) for t in args.tasks.split(",") if t.strip()] if args.tasks else None

    rows: list[tuple[str, list[int]]] = []  # (label, expert sequence)
    for entry in args.routing_logs:
        path_s, label = entry.rsplit(":", 1)
        path = Path(path_s)
        for want in (want_tasks or [None]):
            if path.suffix == ".csv":
                eps = _load_csv(path, want)
            else:
                eps = _load_json(path, want)
            eps = eps[:args.max_episodes_per_label]
            for (t, ep), seq in eps:
                rows.append((f"{label}/T{t}/ep{ep}", seq))

    if not rows:
        raise SystemExit("no rows to plot")

    # Build matrix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    max_chunks = max(len(seq) for _, seq in rows)
    n_experts = max(max(seq) for _, seq in rows) + 1
    mat = np.full((len(rows), max_chunks), -1, dtype=int)
    for i, (_, seq) in enumerate(rows):
        for j, e in enumerate(seq):
            mat[i, j] = e

    cmap = plt.cm.get_cmap("tab20", n_experts)
    fig, ax = plt.subplots(figsize=(0.4 * max_chunks + 2, 0.32 * len(rows) + 1.5))
    for i in range(len(rows)):
        for j in range(max_chunks):
            if mat[i, j] < 0:
                continue
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=cmap(mat[i, j])))
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] >= n_experts // 2 else "black",
                    fontsize=8)
    ax.set_xlim(-0.5, max_chunks - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks(range(max_chunks))
    ax.set_xticklabels([f"c{j}" for j in range(max_chunks)], fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for lab, _ in rows], fontsize=7)
    ax.set_xlabel("action chunk")
    ax.set_title("Top-1 expert per chunk (cell label = expert id)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
