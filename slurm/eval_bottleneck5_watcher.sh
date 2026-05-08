#!/bin/bash
# Watch for new bottleneck5 checkpoints and run a v5-routing eval on each.
# Stops once the 10K-step checkpoint has been evaluated. Designed to run in
# a long-lived tmux session on the eval GPU host.
set -eo pipefail

REPO=/scratch/gpfs/EYSENBACH/ss5822/vla-moe-diversity
CKPT_ROOT=${REPO}/outputs/pi0_moe_bottleneck5/checkpoints
OUT_BASE=${REPO}/outputs/eval
LOG_DIR=${REPO}/logs
mkdir -p "${LOG_DIR}"

export HF_LEROBOT_HOME=/scratch/gpfs/EYSENBACH/ss5822/data
export HF_HOME=/scratch/gpfs/EYSENBACH/ss5822/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /usr/share/Modules/init/bash
module load anaconda3/2025.6
source activate lerobot3

ROBOSUITE_MACROS_PRIVATE=$(python -c "import robosuite, os; print(os.path.join(os.path.dirname(robosuite.__file__), 'macros_private.py'))" 2>/dev/null || true)
if [ -n "${ROBOSUITE_MACROS_PRIVATE}" ] && [ ! -f "${ROBOSUITE_MACROS_PRIVATE}" ]; then
  echo "FILE_LOGGING_LEVEL = None" > "${ROBOSUITE_MACROS_PRIVATE}"
fi

eval_step () {
  local STEP=$1
  local STEP_PADDED
  STEP_PADDED=$(printf "%06d" "${STEP}")
  local CKPT="${CKPT_ROOT}/${STEP_PADDED}/pretrained_model"
  local OUT="${OUT_BASE}/bottleneck5_step$(printf "%03dk" $((STEP/1000)))_v5"
  local LOG="${LOG_DIR}/eval_bottleneck5_step$(printf "%03dk" $((STEP/1000)))_v5_$(date +%Y%m%d_%H%M%S).log"

  echo "[watcher $(date +%H:%M:%S)] Running v5 eval for step ${STEP}"
  echo "[watcher]   CKPT=${CKPT}"
  echo "[watcher]   OUT=${OUT}"
  rm -rf "${OUT}"

  python -m lerobot.scripts.eval_v5_routing \
    --policy.path="${CKPT}" \
    --policy.device=cuda \
    --env.type=libero \
    --env.task=libero_10 \
    --env.task_ids='[0,1,2,3,4,5,6,7,8,9]' \
    --eval.batch_size=1 \
    --eval.n_episodes=1 \
    --output_dir="${OUT}" \
    --job_name=eval_bottleneck5_step${STEP_PADDED}_v5 \
    --seed=1000 \
    '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
    > "${LOG}" 2>&1

  if [ $? -eq 0 ]; then
    echo "[watcher $(date +%H:%M:%S)] Done step ${STEP}; results in ${OUT}"
    echo "[watcher]   pc_success: $(python -c "import json; d=json.load(open('${OUT}/eval_info.json')); print(d['overall']['pc_success'])" 2>/dev/null)"
  else
    echo "[watcher $(date +%H:%M:%S)] FAILED step ${STEP} — see ${LOG}"
  fi
}

cd "${REPO}"

# Steps to evaluate (4K, 6K, 8K, 10K — 2K already evaluated separately).
STEPS_TO_EVAL=(4000 6000 8000 10000)

for STEP in "${STEPS_TO_EVAL[@]}"; do
  STEP_PADDED=$(printf "%06d" "${STEP}")
  CKPT_DIR="${CKPT_ROOT}/${STEP_PADDED}"
  echo "[watcher $(date +%H:%M:%S)] Waiting for ${CKPT_DIR}/pretrained_model to appear ..."

  # Poll until the checkpoint dir + final file are present and stable.
  while true; do
    if [ -f "${CKPT_DIR}/pretrained_model/model.safetensors" ]; then
      # Wait until file size stops changing (write completed).
      SIZE_A=$(stat -c %s "${CKPT_DIR}/pretrained_model/model.safetensors" 2>/dev/null || echo 0)
      sleep 5
      SIZE_B=$(stat -c %s "${CKPT_DIR}/pretrained_model/model.safetensors" 2>/dev/null || echo 0)
      if [ "${SIZE_A}" = "${SIZE_B}" ] && [ "${SIZE_B}" -gt 0 ]; then
        break
      fi
    fi
    sleep 30
  done

  eval_step "${STEP}"
done

echo "[watcher $(date +%H:%M:%S)] All evals complete."
