"""Wrapper around lerobot_eval that captures whole-expert router decisions.

Installs a forward hook on `policy.model.whole_expert_router` and tags every
top-k routing decision with the (task_group, task_id) currently being
evaluated (set via a monkey-patched `run_one`). Dumps routing_log.json and
a per-task expert-frequency heatmap PNG/CSV alongside the standard eval
artifacts under `--output_dir`.

Usage matches lerobot_eval (same CLI flags). Output:
  <output_dir>/routing_log.json     raw per-call records
  <output_dir>/expert_freq.csv      task x expert top-1 frequency
  <output_dir>/expert_freq.png      heatmap (matplotlib, no seaborn)
"""

from __future__ import annotations

import atexit
import json
import sys
from collections import defaultdict
from pathlib import Path

import lerobot.scripts.lerobot_eval as lev


class RoutingLogger:
    def __init__(self):
        self.records: list[dict] = []
        self.current = ("", -1)
        self.installed = False

    def set_task(self, group: str, task_id: int):
        self.current = (group, int(task_id))

    def hook(self, _module, _inputs, outputs):
        # WholeExpertRouter forward returns (indices, weights, aux_dict)
        idx, wts, _ = outputs
        g, t = self.current
        self.records.append(
            {
                "group": g,
                "task": t,
                "indices": idx.detach().cpu().tolist(),
                "weights": wts.detach().to("cpu", dtype=__import__("torch").float32).tolist(),
            }
        )


R = RoutingLogger()


def _output_dir_from_argv() -> Path | None:
    for i, a in enumerate(sys.argv):
        if a.startswith("--output_dir="):
            return Path(a.split("=", 1)[1])
        if a == "--output_dir" and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1])
    return None


_OUTPUT_DIR: Path | None = _output_dir_from_argv()


_orig_run_one = lev.run_one


def _find_router(pol):
    """Walk the policy wrapper chain to find ``whole_expert_router``.

    PEFT-wrapped policies have an extra layer of indirection: ``pol.model``
    is the inner ``PI0Policy`` (not ``PI0Pytorch``) so
    ``pol.model.whole_expert_router`` is None. Traverse via ``.model``
    attributes until we find one that has the router.
    """
    seen = set()
    cur = pol
    for _ in range(6):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        router = getattr(cur, "whole_expert_router", None)
        if router is not None:
            return router
        cur = getattr(cur, "model", None)
    return None


def _patched_run_one(task_group, task_id, env, **kw):
    R.set_task(task_group, task_id)
    pol = kw["policy"]
    if not R.installed:
        router = _find_router(pol)
        if router is None:
            raise RuntimeError(
                "Could not locate whole_expert_router on the policy — this script is for v5 (whole-expert) checkpoints only."
            )
        router.register_forward_hook(R.hook)
        R.installed = True
    return _orig_run_one(task_group, task_id, env, **kw)


lev.run_one = _patched_run_one


def _dump():
    if _OUTPUT_DIR is None or not R.records:
        print(f"[routing] nothing to dump (records={len(R.records)}, dir={_OUTPUT_DIR})")
        return
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = _OUTPUT_DIR / "routing_log.json"
    with open(raw, "w") as f:
        json.dump(R.records, f)
    print(f"[routing] wrote {len(R.records)} records to {raw}")

    # Aggregate top-1 expert frequency per (group, task).
    counts: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0] * 64)  # 64 is upper bound
    n_experts_seen = 0
    for rec in R.records:
        for sample in rec["indices"]:  # sample is [k] (top-k indices)
            top1 = int(sample[0])
            counts[(rec["group"], rec["task"])][top1] += 1
            n_experts_seen = max(n_experts_seen, top1 + 1)

    keys = sorted(counts.keys())
    csv_path = _OUTPUT_DIR / "expert_freq.csv"
    with open(csv_path, "w") as f:
        f.write("group,task," + ",".join(f"expert_{i}" for i in range(n_experts_seen)) + "\n")
        for g, t in keys:
            row = counts[(g, t)][:n_experts_seen]
            tot = sum(row) or 1
            f.write(f"{g},{t}," + ",".join(f"{c / tot:.4f}" for c in row) + "\n")
    print(f"[routing] wrote per-task top-1 frequencies to {csv_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        mat = np.array([[counts[(g, t)][i] for i in range(n_experts_seen)] for g, t in keys], dtype=float)
        mat = mat / mat.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(0.4 * n_experts_seen + 2, 0.3 * len(keys) + 2))
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(n_experts_seen))
        ax.set_xticklabels([f"e{i}" for i in range(n_experts_seen)], rotation=90, fontsize=8)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([f"{g}/t{t}" for g, t in keys], fontsize=8)
        ax.set_xlabel("expert (top-1)")
        ax.set_title("v5 expert routing frequency per task")
        fig.colorbar(im, ax=ax, fraction=0.02)
        fig.tight_layout()
        png = _OUTPUT_DIR / "expert_freq.png"
        fig.savefig(png, dpi=120)
        plt.close(fig)
        print(f"[routing] wrote heatmap to {png}")
    except Exception as e:
        print(f"[routing] heatmap skipped: {e}")


atexit.register(_dump)


if __name__ == "__main__":
    # Run lerobot_eval's parser-wrapped entrypoint as-is. Our monkey-patch on
    # `run_one` will fire from within it, our atexit handler will dump.
    lev.eval_main()
