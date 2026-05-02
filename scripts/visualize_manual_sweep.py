"""Aggregate a manual-routing sweep into pivoted markdown + heatmap PNG.

Reads ``sweep_summary.csv`` (produced by ``run_manual_sweep.py``) and emits:
  - ``sweep_pivot.md`` — markdown table, rows=condition, cols=task_id, cells=success_rate
  - ``sweep_pivot.png`` — heatmap of the same pivot

Usage:
  python scripts/visualize_manual_sweep.py outputs/evals/manual_routing/<sweep_root>
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_root", type=Path)
    ap.add_argument("--row_order", default="",
                    help="comma-separated condition prefixes for sort order")
    args = ap.parse_args()

    csv_path = args.sweep_root / "sweep_summary.csv"
    if not csv_path.exists():
        raise SystemExit(f"missing {csv_path}")

    rows = list(csv.DictReader(open(csv_path)))
    # pivot: condition -> {task_id: (n_succ, n_ep)}
    pivot: dict[str, dict[int, tuple[int, int]]] = defaultdict(dict)
    for r in rows:
        cond = r["condition"]
        tid = int(r["task_id"])
        pivot[cond][tid] = (int(r["n_success"]), int(r["n_episodes"]))

    all_tasks = sorted({t for d in pivot.values() for t in d})
    conds = sorted(pivot.keys())
    if args.row_order:
        order = [s.strip() for s in args.row_order.split(",") if s.strip()]
        def _key(c: str) -> tuple:
            for i, p in enumerate(order):
                if c.startswith(p):
                    return (i, c)
            return (len(order), c)
        conds.sort(key=_key)

    # Markdown
    md_lines: list[str] = []
    md_lines.append("# Manual-routing sweep — success-rate pivot\n")
    md_lines.append(f"Source: `{csv_path}`\n")
    header = "| condition | " + " | ".join(f"T{t}" for t in all_tasks) + " | mean |"
    sep = "|" + "|".join(["---"] * (len(all_tasks) + 2)) + "|"
    md_lines.append(header)
    md_lines.append(sep)
    for cond in conds:
        cells = []
        rates = []
        for t in all_tasks:
            if t in pivot[cond]:
                ns, ne = pivot[cond][t]
                rate = ns / ne if ne else 0.0
                cells.append(f"{ns}/{ne} ({rate:.2f})")
                rates.append(rate)
            else:
                cells.append("—")
        mean = sum(rates) / len(rates) if rates else 0.0
        md_lines.append(f"| `{cond}` | " + " | ".join(cells) + f" | {mean:.2f} |")
    md_lines.append("")

    # Per-task best
    md_lines.append("## Best per task\n")
    for t in all_tasks:
        best = max(
            ((cond, pivot[cond][t]) for cond in conds if t in pivot[cond]),
            key=lambda kv: kv[1][0] / max(kv[1][1], 1),
            default=None,
        )
        if best is None:
            continue
        cond, (ns, ne) = best
        md_lines.append(f"- **T{t}**: `{cond}` with {ns}/{ne} = {ns / max(ne, 1):.2f}")
    md_lines.append("")

    out_md = args.sweep_root / "sweep_pivot.md"
    out_md.write_text("\n".join(md_lines))
    print(f"wrote {out_md}")

    # Heatmap (optional, skip if matplotlib missing)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        mat = np.full((len(conds), len(all_tasks)), np.nan)
        for i, c in enumerate(conds):
            for j, t in enumerate(all_tasks):
                if t in pivot[c]:
                    ns, ne = pivot[c][t]
                    mat[i, j] = ns / ne if ne else 0.0
        fig, ax = plt.subplots(figsize=(0.6 * len(all_tasks) + 3, 0.32 * len(conds) + 1.5))
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        for i in range(len(conds)):
            for j in range(len(all_tasks)):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.5 else "black", fontsize=7)
        ax.set_xticks(range(len(all_tasks)))
        ax.set_xticklabels([f"T{t}" for t in all_tasks])
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels(conds, fontsize=7)
        ax.set_xlabel("task")
        ax.set_title(f"manual-routing success rate ({args.sweep_root.name})")
        fig.colorbar(im, ax=ax, fraction=0.025)
        fig.tight_layout()
        out_png = args.sweep_root / "sweep_pivot.png"
        fig.savefig(out_png, dpi=130)
        plt.close(fig)
        print(f"wrote {out_png}")
    except Exception as e:
        print(f"heatmap skipped: {e}")


if __name__ == "__main__":
    main()
