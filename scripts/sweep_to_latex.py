"""Convert sweep_summary.csv into a paper-ready LaTeX table."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


# Map raw condition prefixes to row labels and a coarse category for sorting/styling.
CATEGORY = [
    ("passthrough", "passthrough (learned router)"),
    ("replay", "replay (recorded success)"),
    ("phase_t5_approach", "phase $e_8 \\to e_1$"),
    ("phase_t5_reversed", "phase reversed $e_1 \\to e_8$"),
    ("phase_t2_approach", "phase $e_8 \\to (e_1, e_9, e_5)$"),
    ("phase_t2_reversed", "phase reversed"),
    ("phase_t5_on", "T5 phase transferred"),
    ("single_e1", "single expert $e_1$"),
    ("single_e8", "single expert $e_8$"),
    ("single_e9", "single expert $e_9$"),
    ("single_e5", "single expert $e_5$"),
    ("single_e11", "single expert $e_{11}$"),
    ("single_e15", "single expert $e_{15}$"),
    ("single_e3", "single expert $e_3$ (rare)"),
    ("random", "random top-2 (control)"),
]


def _label_and_task(name: str, task_id: int, env_task: str) -> tuple[str, str]:
    """Return (row_label, column_label) for a (condition_name, task_id)."""
    for prefix, lab in CATEGORY:
        if name.startswith(prefix):
            return lab, f"{env_task}/T{task_id}"
    return name, f"{env_task}/T{task_id}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_csv", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--env_task_default", default="libero_10")
    ap.add_argument("--bold_max_per_col", action="store_true",
                    help="Bold the maximum success rate in each column")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.sweep_csv)))
    # Determine env_task for each row from the eval_info if available — for
    # now infer from condition name suffix (e.g. "*_libero_goal_t1") otherwise
    # default. Easier: read env_task from each per-condition manual_routing_manifest.
    # Fallback: assume args.env_task_default.

    # Build row_label -> {col_label: success_rate}
    pivot: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    cond_env_task: dict[str, str] = {}
    sweep_root = args.sweep_csv.parent
    for r in rows:
        cond = r["condition"]
        # Infer env_task from condition name (we encoded it in family3)
        env_task = args.env_task_default
        m = re.search(r"libero_(goal|object|spatial)", cond)
        if m:
            env_task = "libero_" + m.group(1)
        elif "libero_10" in cond:
            env_task = "libero_10"
        cond_env_task[cond] = env_task

        task_id = int(r["task_id"])
        row_label, col_label = _label_and_task(cond, task_id, env_task)
        pivot[row_label][col_label] = (int(r["n_success"]), int(r["n_episodes"]))

    # Order rows by CATEGORY definition; cols sorted alphabetically.
    row_order = []
    for prefix, lab in CATEGORY:
        if lab in pivot:
            row_order.append(lab)
    # Add any that didn't match.
    for lab in pivot:
        if lab not in row_order:
            row_order.append(lab)
    cols = sorted({c for d in pivot.values() for c in d})

    # Build LaTeX
    n_cols = len(cols)
    out = []
    out.append(r"\begin{table}[h]")
    out.append(r"\centering")
    out.append(r"\caption{\textbf{Manual-routing success rate.} Each cell is "
               r"successes/episodes from one batch of 5 episodes at seed 1000.}")
    out.append(r"\label{tab:manual_routing}")
    out.append(r"\begin{tabular}{l" + "c" * n_cols + r"}")
    out.append(r"\toprule")
    out.append(r"\textbf{Condition} & " + " & ".join(c.replace("_", r"\_") for c in cols) + r" \\")
    out.append(r"\midrule")
    # For each column, find max success rate for bolding
    col_max_rate = {}
    for c in cols:
        rates = [(pivot[r][c][0] / max(pivot[r][c][1], 1)) for r in row_order if c in pivot[r]]
        col_max_rate[c] = max(rates) if rates else 0.0

    for r_label in row_order:
        cells = []
        for c in cols:
            if c in pivot[r_label]:
                ns, ne = pivot[r_label][c]
                rate = ns / max(ne, 1)
                txt = f"{ns}/{ne}"
                if args.bold_max_per_col and rate >= col_max_rate[c] - 1e-9 and rate > 0:
                    txt = r"\textbf{" + txt + r"}"
            else:
                txt = "---"
            cells.append(txt)
        out.append(f"{r_label} & " + " & ".join(cells) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")

    text = "\n".join(out) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
