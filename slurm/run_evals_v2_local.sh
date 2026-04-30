#!/bin/bash
# Run all 8 v2 LIBERO evals directly on della-fong (2x A100), no slurm.
# GPU 0 handles baseline; GPU 1 handles moe_diversity. Each runs its 4 suites sequentially.
set -uo pipefail

REPO=/scratch/gpfs/FHEIDE/rj2807/vla-moe-diversity
LOGDIR=/scratch/gpfs/FHEIDE/rj2807/logs
mkdir -p "$LOGDIR"

export HF_LEROBOT_HOME=/scratch/gpfs/FHEIDE/rj2807/lerobot_data
export HF_HOME=/scratch/gpfs/FHEIDE/rj2807/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH=/scratch/gpfs/FHEIDE/rj2807/cache/libero

source "$REPO/.venv/bin/activate"

ROBOSUITE_MACROS_PRIVATE=$(python -c "import robosuite; import os; print(os.path.join(os.path.dirname(robosuite.__file__), 'macros_private.py'))" 2>/dev/null || true)
if [ -n "$ROBOSUITE_MACROS_PRIVATE" ] && [ ! -f "$ROBOSUITE_MACROS_PRIVATE" ]; then
  echo "FILE_LOGGING_LEVEL = None" > "$ROBOSUITE_MACROS_PRIVATE"
fi

RENAME_MAP='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'

run_one() {
  local gpu=$1 variant=$2 task=$3
  local tag="eval_v2_${variant}_${task}"
  local log="$LOGDIR/${tag}_local.log"
  echo "[$(date +%H:%M:%S)] GPU${gpu} START ${tag}" | tee -a "$LOGDIR/run_evals_v2_local.main.log"
  CUDA_VISIBLE_DEVICES=$gpu python -m lerobot.scripts.lerobot_eval \
    --policy.path=/scratch/gpfs/FHEIDE/rj2807/outputs/pi0_${variant}_v2/checkpoints/020000/pretrained_model \
    --env.type=libero \
    --env.task=${task} \
    --eval.batch_size=5 \
    --eval.n_episodes=50 \
    --policy.device=cuda \
    --output_dir=/scratch/gpfs/FHEIDE/rj2807/outputs/evals_v2/${variant}/${task} \
    --job_name=${tag} \
    --seed=1000 \
    "--rename_map=${RENAME_MAP}" \
    > "$log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] GPU${gpu} END   ${tag} rc=${rc}" | tee -a "$LOGDIR/run_evals_v2_local.main.log"
}

worker() {
  local gpu=$1 variant=$2
  for task in libero_10 libero_goal libero_object libero_spatial; do
    run_one "$gpu" "$variant" "$task"
  done
}

worker 0 baseline &
PID0=$!
worker 1 moe_diversity &
PID1=$!

wait "$PID0" "$PID1"
echo "[$(date +%H:%M:%S)] all workers done" | tee -a "$LOGDIR/run_evals_v2_local.main.log"
