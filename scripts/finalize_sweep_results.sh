#!/usr/bin/env bash
# Finalize manual-routing sweep results: regenerate sweep_summary.csv,
# build markdown pivots, and emit paper-ready LaTeX tables.
#
# Usage: ./scripts/finalize_sweep_results.sh
#
# Outputs (per sweep):
#   <sweep_root>/sweep_summary.csv        — flat per-(condition, task) CSV
#   <sweep_root>/sweep_pivot.md           — markdown pivot
#   <sweep_root>/sweep_pivot.png          — heatmap PNG
#   tables/{family1,family2,family3}.tex  — LaTeX tables for paper
set -uo pipefail

REPO=/scratch/gpfs/FHEIDE/rj2807/vla-moe-diversity
OUT=/scratch/gpfs/FHEIDE/rj2807/outputs/evals/manual_routing
TABLES=$REPO/tables
mkdir -p $TABLES

cd $REPO
source .venv/bin/activate

for s in family1_essential family2_transfer family3_ood validation; do
  if [ -d "$OUT/$s" ]; then
    echo "=== $s ==="
    # Re-aggregate (in case the sweep was interrupted)
    python scripts/run_manual_sweep.py \
      --sweep_root="$OUT/$s" \
      --conditions_json="$REPO/sweeps/$s.json" \
      --aggregate_only 2>/dev/null || \
      python -c "
from pathlib import Path
import sys
sys.path.insert(0, '$REPO')
from scripts.run_manual_sweep import _aggregate
_aggregate(Path('$OUT/$s'))
" 2>/dev/null

    if [ -f "$OUT/$s/sweep_summary.csv" ]; then
      python scripts/visualize_manual_sweep.py "$OUT/$s" \
        --row_order=passthrough,replay,phase,single,random
      python scripts/sweep_to_latex.py "$OUT/$s/sweep_summary.csv" \
        --out="$TABLES/${s}.tex" --bold_max_per_col
    fi
  fi
done

echo "Done. Tables in $TABLES, pivots in each $OUT/<sweep>/sweep_pivot.{md,png}"
