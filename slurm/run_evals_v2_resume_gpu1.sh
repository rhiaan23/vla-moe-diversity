#!/bin/bash
# Resume v2 eval on GPU 1 only (GPU 0 is training v3).
# Skips libero_10 (already complete rc=0 for both variants).
# Runs 6 remaining: {baseline, moe_diversity} x {goal, object, spatial}.
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
export CUDA_VISIBLE_DEVICES=1

source "$REPO/.venv/bin/activate"

ROBOSUITE_MACROS_PRIVATE=$(python -c "import robosuite; import os; print(os.path.join(os.path.dirname(robosuite.__file__), 'macros_private.py'))" 2>/dev/null || true)
if [ -n "$ROBOSUITE_MACROS_PRIVATE" ] && [ ! -f "$ROBOSUITE_MACROS_PRIVATE" ]; then
  echo "FILE_LOGGING_LEVEL = None" > "$ROBOSUITE_MACROS_PRIVATE"
fi

RENAME_MAP='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
MAIN_LOG="$LOGDIR/run_evals_v2_resume_gpu1.main.log"

run_one() {
  local variant=$1 task=$2
  local tag="eval_v2_${variant}_${task}"
  local log="$LOGDIR/${tag}_local.log"
  if [ -f "/scratch/gpfs/FHEIDE/rj2807/outputs/evals_v2/${variant}/${task}/eval_info.json" ]; then
    echo "[$(date +%H:%M:%S)] SKIP ${tag} (already complete)" | tee -a "$MAIN_LOG"
    return
  fi
  rm -rf "/scratch/gpfs/FHEIDE/rj2807/outputs/evals_v2/${variant}/${task}"
  echo "[$(date +%H:%M:%S)] START ${tag}" | tee -a "$MAIN_LOG"
  python -m lerobot.scripts.lerobot_eval \
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
  echo "[$(date +%H:%M:%S)] END   ${tag} rc=${rc}" | tee -a "$MAIN_LOG"
}

for task in libero_goal libero_object libero_spatial; do
  for variant in baseline moe_diversity; do
    run_one "$variant" "$task"
  done
done

echo "[$(date +%H:%M:%S)] resume done" | tee -a "$MAIN_LOG"
