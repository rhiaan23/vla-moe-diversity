#!/usr/bin/env bash
# Run all manual-routing sweeps in series (one GPU).
# Each sweep is independent; we run sequentially because they all use the
# same H100 and would compete for memory.

set -uo pipefail

REPO=/scratch/gpfs/FHEIDE/rj2807/vla-moe-diversity
OUT=/scratch/gpfs/FHEIDE/rj2807/outputs/evals/manual_routing

cd "$REPO"
source .venv/bin/activate

ts() { date '+%H:%M:%S'; }

run_sweep() {
  local cfg="$1"
  local root="$2"
  echo "============================================================"
  echo "[$(ts)] starting $(basename "$cfg") -> $root"
  echo "============================================================"
  python scripts/run_manual_sweep.py \
    --sweep_root="$root" \
    --conditions_json="$cfg" \
    --batch_size=5 --n_episodes=5 --seed=1000 \
    --skip_existing
  rc=$?
  echo "[$(ts)] finished $(basename "$cfg") rc=$rc"
}

run_sweep sweeps/family1_essential.json "$OUT/family1_essential"
run_sweep sweeps/family2_transfer.json "$OUT/family2_transfer"
run_sweep sweeps/family3_ood.json "$OUT/family3_ood"

# Final aggregation pivot tables for each
for s in family1_essential family2_transfer family3_ood; do
  if [ -f "$OUT/$s/sweep_summary.csv" ]; then
    python scripts/visualize_manual_sweep.py "$OUT/$s" \
      --row_order=passthrough,replay,phase,single,random
  fi
done

echo "[$(ts)] all sweeps done"
