"""Build standard manual-routing programs for the compositional generality sweep.

Generates JSON files under ``programs/`` for:
  - single-expert-N (no by_chunk; default = {[N], [1.0]})
  - random-per-chunk template (a marker file; the eval script's --manual_mode=random handles this)
  - reversed (read a phase program, reverse its by_chunk)
  - replay (extract per-episode router sequence from chunk_routing.csv)

Use cases (driven by the project plan):
  1. Single-expert sweep across active experts {1, 5, 8, 9, 11, 15} and a
     few controls {3, 7, 13}.
  2. Replay successful router episodes — known-good sequences.
  3. Reversed control: take a phase program, reverse it; if order matters,
     reversed should fail.

Usage:
  python scripts/build_programs.py single_expert --experts 1,5,8,9,11,15,3,7,13
  python scripts/build_programs.py replay_all_success \\
      outputs/evals/v5_libero_10 --tasks 2,3,5
  python scripts/build_programs.py reverse \\
      programs/t6_phase_v1.json programs/t6_phase_v1_reversed.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

PROGRAMS_DIR = Path(__file__).resolve().parent.parent / "programs"


# ---------------------------------------------------------------------------

def cmd_single_expert(args: argparse.Namespace) -> None:
    """Write programs/single_expert_e<N>.json for each expert id."""
    experts = [int(e) for e in args.experts.split(",")]
    PROGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    for e in experts:
        p = {
            "kind": "single_expert",
            "expert_id": e,
            "default": {"indices": [e], "weights": [1.0]},
        }
        out = PROGRAMS_DIR / f"single_expert_e{e}.json"
        with open(out, "w") as f:
            json.dump(p, f, indent=2)
        print(f"wrote {out}")


# ---------------------------------------------------------------------------

def cmd_replay_all_success(args: argparse.Namespace) -> None:
    """For each successful episode in chunk_routing.csv, dump a replay program."""
    eval_dir = Path(args.eval_dir)
    chunk_csv = eval_dir / "chunk_routing.csv"
    run_csv = eval_dir / "run_summary.csv"
    if not chunk_csv.exists() or not run_csv.exists():
        raise SystemExit(f"missing chunk_routing.csv or run_summary.csv under {eval_dir}")

    # Identify successful episodes.
    succ_eps: dict[int, list[int]] = defaultdict(list)
    with open(run_csv) as f:
        for row in csv.DictReader(f):
            if int(row["success"]) == 1:
                succ_eps[int(row["task"])].append(int(row["episode"]))

    # Optional task filter.
    if args.tasks:
        wanted = {int(t) for t in args.tasks.split(",")}
        succ_eps = {t: eps for t, eps in succ_eps.items() if t in wanted}

    if not succ_eps:
        print("no successful episodes found")
        return

    # Group chunk rows.
    chunks_by_te: dict[tuple[int, int], list[dict]] = defaultdict(list)
    with open(chunk_csv) as f:
        for row in csv.DictReader(f):
            chunks_by_te[(int(row["task"]), int(row["episode"]))].append(row)

    PROGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for task, eps in sorted(succ_eps.items()):
        for ep in eps:
            rows = sorted(chunks_by_te[(task, ep)], key=lambda r: int(r["chunk"]))
            by_chunk = [
                {
                    "indices": [int(r["top1_expert"]), int(r["top2_expert"])],
                    "weights": [float(r["top1_weight"]), float(r["top2_weight"])],
                }
                for r in rows
            ]
            program = {
                "kind": "replay_success",
                "source_eval_dir": str(eval_dir),
                "source_task": task,
                "source_episode": ep,
                "n_chunks": len(by_chunk),
                "by_chunk": by_chunk,
                "default": by_chunk[-1] if by_chunk else {"indices": [0], "weights": [1.0]},
            }
            out = PROGRAMS_DIR / f"replay_success_t{task}_ep{ep}.json"
            with open(out, "w") as f:
                json.dump(program, f, indent=2)
            written += 1
    print(f"wrote {written} replay-success programs to {PROGRAMS_DIR}")


# ---------------------------------------------------------------------------

def cmd_reverse(args: argparse.Namespace) -> None:
    src = json.load(open(args.src))
    by_chunk = list(src.get("by_chunk", []))
    by_chunk.reverse()
    out = {
        "kind": "reversed",
        "source": str(args.src),
        "by_chunk": by_chunk,
        "default": src.get("default", {"indices": [0], "weights": [1.0]}),
    }
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    with open(args.dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.dst} ({len(by_chunk)} chunks reversed)")


# ---------------------------------------------------------------------------

def cmd_phase(args: argparse.Namespace) -> None:
    """Build a phase program: a list of (n_chunks, [experts], [weights]) phases.

    Spec (CLI): --spec='2:11/1.0;3:5,9/0.5,0.5;2:8/1.0' means:
      chunks 0-1: expert 11 (weight 1.0)
      chunks 2-4: experts 5,9 (weights 0.5, 0.5)
      chunks 5-6: expert 8 (weight 1.0)
    """
    by_chunk = []
    parts = [p for p in args.spec.split(";") if p]
    for part in parts:
        n_str, ew = part.split(":")
        n = int(n_str)
        if "/" in ew:
            es, ws = ew.split("/")
        else:
            es, ws = ew, ",".join(["1.0"] * len(ew.split(",")))
        indices = [int(e) for e in es.split(",")]
        weights = [float(w) for w in ws.split(",")]
        if len(indices) != len(weights):
            raise SystemExit(f"phase spec {part}: indices/weights length mismatch")
        for _ in range(n):
            by_chunk.append({"indices": indices, "weights": weights})
    out = {
        "kind": "phase",
        "spec": args.spec,
        "n_chunks": len(by_chunk),
        "by_chunk": by_chunk,
        "default": by_chunk[-1] if by_chunk else {"indices": [0], "weights": [1.0]},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out} ({len(by_chunk)} chunks)")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    se = sp.add_parser("single_expert")
    se.add_argument("--experts", default="1,5,8,9,11,15,3,7,13")
    se.set_defaults(func=cmd_single_expert)

    rp = sp.add_parser("replay_all_success")
    rp.add_argument("eval_dir")
    rp.add_argument("--tasks", default="", help="comma-separated task ids; empty = all")
    rp.set_defaults(func=cmd_replay_all_success)

    rv = sp.add_parser("reverse")
    rv.add_argument("src")
    rv.add_argument("dst")
    rv.set_defaults(func=cmd_reverse)

    ph = sp.add_parser("phase")
    ph.add_argument("--spec", required=True, help="e.g. '2:11/1.0;3:5,9/0.5,0.5;2:8/1.0'")
    ph.add_argument("--out", required=True)
    ph.set_defaults(func=cmd_phase)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
