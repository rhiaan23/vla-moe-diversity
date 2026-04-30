#!/bin/bash
# Submits all 8 v2 eval jobs (2 variants x 4 LIBERO suites).
# Usage: bash slurm/submit_evals_v2.sh
set -euo pipefail
cd "$(dirname "$0")"

for s in \
  eval_v2_baseline_libero10.slurm \
  eval_v2_baseline_goal.slurm \
  eval_v2_baseline_object.slurm \
  eval_v2_baseline_spatial.slurm \
  eval_v2_moe_diversity_libero10.slurm \
  eval_v2_moe_diversity_goal.slurm \
  eval_v2_moe_diversity_object.slurm \
  eval_v2_moe_diversity_spatial.slurm; do
  sbatch "$s"
done

echo
squeue -u "$USER"
