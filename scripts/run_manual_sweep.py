"""Run a sweep of manual-routing eval conditions and aggregate results.

The "sweep" is a list of conditions, each spawning one ``eval_v5_manual_routing``
invocation. After all conditions finish, the script reads each ``eval_info.json``
and emits a single CSV ``sweep_summary.csv`` under the sweep root with one row
per (condition, task_id).

Usage:
  python scripts/run_manual_sweep.py \\
    --sweep_root=/scratch/gpfs/FHEIDE/rj2807/outputs/evals/manual_routing/single_expert_t235 \\
    --conditions_json=sweeps/single_expert_t235.json \\
    --policy_path=/scratch/gpfs/FHEIDE/rj2807/outputs/pi0_moe_whole_v5_h100/checkpoints/last/pretrained_model \\
    --batch_size=5 --n_episodes=5 --seed=1000

Conditions JSON schema:
  [
    {
      "name": "single_e1_t5",
      "manual_mode": "scripted",
      "manual_program": "programs/single_expert_e1.json",
      "env_task": "libero_10",
      "task_ids": [5]
    },
    {
      "name": "random_t5",
      "manual_mode": "random",
      "manual_random_seed": 42,
      "env_task": "libero_10",
      "task_ids": [5]
    },
    ...
  ]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _build_cmd(cond: dict, outdir: Path, args: argparse.Namespace) -> list[str]:
    cmd = [
        "python", "-m", "scripts.eval_v5_manual_routing",
        f"--manual_mode={cond['manual_mode']}",
        f"--policy.path={args.policy_path}",
        f"--policy.moe_top_k={args.moe_top_k}",
        f"--env.type=libero",
        f"--env.task={cond['env_task']}",
        f"--env.task_ids={json.dumps(cond['task_ids'])}",
        f"--eval.batch_size={args.batch_size}",
        f"--eval.n_episodes={args.n_episodes}",
        f"--policy.device=cuda",
        f"--output_dir={outdir}",
        f"--job_name={cond['name']}",
        f"--seed={args.seed}",
        '--rename_map={"observation.images.image": "observation.images.camera1", '
        '"observation.images.image2": "observation.images.camera2"}',
    ]
    if cond["manual_mode"] == "scripted":
        if "manual_program" not in cond:
            raise SystemExit(f"condition {cond['name']}: scripted requires manual_program")
        cmd.append(f"--manual_program={cond['manual_program']}")
    if "manual_random_seed" in cond:
        cmd.append(f"--manual_random_seed={cond['manual_random_seed']}")
    return cmd


def _aggregate(sweep_root: Path) -> Path:
    rows = []
    for cond_dir in sorted(p for p in sweep_root.iterdir() if p.is_dir()):
        ei = cond_dir / "eval_info.json"
        if not ei.exists():
            continue
        d = json.load(open(ei))
        per_task = d.get("per_task", [])
        for entry in per_task:
            metrics = entry.get("metrics", {})
            successes = metrics.get("successes", [])
            n = len(successes)
            n_succ = sum(1 for s in successes if s)
            rows.append(
                {
                    "condition": cond_dir.name,
                    "task_group": entry.get("task_group", ""),
                    "task_id": entry.get("task_id", -1),
                    "n_episodes": n,
                    "n_success": n_succ,
                    "success_rate": (n_succ / n) if n else 0.0,
                    "successes": ",".join("1" if s else "0" for s in successes),
                }
            )
    if not rows:
        print("[aggregate] no eval_info.json files found")
        return None
    out = sweep_root / "sweep_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[aggregate] wrote {len(rows)} rows to {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", required=True, type=Path)
    ap.add_argument("--conditions_json", required=True, type=Path)
    ap.add_argument(
        "--policy_path",
        default="/scratch/gpfs/FHEIDE/rj2807/outputs/pi0_moe_whole_v5_h100/checkpoints/last/pretrained_model",
    )
    ap.add_argument("--moe_top_k", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--n_episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip conditions whose output dir already has eval_info.json")
    ap.add_argument("--aggregate_only", action="store_true",
                    help="Skip running anything; just rebuild sweep_summary.csv from existing dirs")
    args = ap.parse_args()

    args.sweep_root.mkdir(parents=True, exist_ok=True)
    conditions = json.load(open(args.conditions_json))
    if not isinstance(conditions, list):
        raise SystemExit("conditions JSON must be a list")

    if args.aggregate_only:
        _aggregate(args.sweep_root)
        return

    env = os.environ.copy()
    env.setdefault("HF_LEROBOT_HOME", "/scratch/gpfs/FHEIDE/rj2807/lerobot_data")
    env.setdefault("HF_HOME", "/scratch/gpfs/FHEIDE/rj2807/cache/huggingface")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("MUJOCO_GL", "egl")

    start_t = time.time()
    for i, cond in enumerate(conditions, 1):
        outdir = args.sweep_root / cond["name"]
        ei = outdir / "eval_info.json"
        if args.skip_existing and ei.exists():
            print(f"[{i}/{len(conditions)}] SKIP {cond['name']} (eval_info.json exists)")
            continue
        outdir.mkdir(parents=True, exist_ok=True)
        # Wipe stale routing artifacts but keep skeleton.
        for stale in ("routing_log.json", "manual_routing_manifest.json", "manual_program_used.json"):
            (outdir / stale).unlink(missing_ok=True)

        cmd = _build_cmd(cond, outdir, args)
        log = outdir / "stdout.log"
        t0 = time.time()
        print(f"[{i}/{len(conditions)}] {cond['name']}  ->  {outdir}")
        print(f"  cmd: {' '.join(shlex.quote(c) for c in cmd)}")
        with open(log, "wb") as lf:
            rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        status = "OK" if rc == 0 else f"FAIL rc={rc}"
        print(f"  -> {status}  in {dt:.1f}s   (log: {log})")
        if rc != 0:
            print(f"  see {log} for details; continuing")

    print(f"\nsweep done in {time.time() - start_t:.1f}s")
    _aggregate(args.sweep_root)


if __name__ == "__main__":
    main()
