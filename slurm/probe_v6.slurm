#!/bin/bash
#SBATCH --job-name=probe-v6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/gpfs/FHEIDE/rj2807/logs/probe_v6_%x_%j.out
#SBATCH --error=/scratch/gpfs/FHEIDE/rj2807/logs/probe_v6_%x_%j.err
#
# Routing probe wrapper. Pass the run name and (optional) checkpoint step as env vars.
# Examples:
#   sbatch -J probe-v6-div   --export=ALL,RUN=pi0_moe_v6_diversity_h100,STEP=last slurm/probe_v6.slurm
#   sbatch -J probe-v6-noor  --export=ALL,RUN=pi0_moe_v6_no_orth_h100,STEP=002000 slurm/probe_v6.slurm

set -uo pipefail

RUN=${RUN:?must set RUN env var, e.g. pi0_moe_v6_diversity_h100}
STEP=${STEP:-last}
N_EP=${N_EP:-10}

CKPT=/scratch/gpfs/FHEIDE/rj2807/outputs/${RUN}/checkpoints/${STEP}/pretrained_model
OUT=/scratch/gpfs/FHEIDE/rj2807/outputs/evals_${RUN}_routing_${STEP}

if [ ! -d "$CKPT" ]; then
  echo "[probe_v6] checkpoint missing: $CKPT" >&2
  ls /scratch/gpfs/FHEIDE/rj2807/outputs/${RUN}/checkpoints/ >&2
  exit 1
fi

mkdir -p "$OUT"

export HF_LEROBOT_HOME=/scratch/gpfs/FHEIDE/rj2807/lerobot_data
export HF_HOME=/scratch/gpfs/FHEIDE/rj2807/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl

source /scratch/gpfs/FHEIDE/rj2807/vla-moe-diversity/.venv/bin/activate

ROBOSUITE_MACROS_PRIVATE=$(python -c "import robosuite, os; print(os.path.join(os.path.dirname(robosuite.__file__), 'macros_private.py'))" 2>/dev/null || true)
if [ -n "$ROBOSUITE_MACROS_PRIVATE" ] && [ ! -f "$ROBOSUITE_MACROS_PRIVATE" ]; then
  echo "FILE_LOGGING_LEVEL = None" > "$ROBOSUITE_MACROS_PRIVATE"
fi

cd /scratch/gpfs/FHEIDE/rj2807

echo "[probe_v6] RUN=$RUN STEP=$STEP CKPT=$CKPT OUT=$OUT"
python /scratch/gpfs/FHEIDE/rj2807/logs/probe_routing.py "$CKPT" "$OUT" "$N_EP"
RC=$?
if [ $RC -ne 0 ]; then
  echo "[probe_v6] probe_routing.py exited $RC" >&2
  exit $RC
fi

if [ -f "$OUT/routing.npz" ]; then
  python /scratch/gpfs/FHEIDE/rj2807/logs/analyze_routing_param.py \
    "$OUT/routing.npz" "$OUT/routing_report.md" "${RUN}_${STEP}"
else
  echo "[probe_v6] WARNING: $OUT/routing.npz not produced — skip analyze" >&2
  exit 2
fi
