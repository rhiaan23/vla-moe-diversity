#!/bin/bash
# Two-phase replay test: record per-chunk (z, router_idx, router_wts) for one
# rollout of LIBERO task 0 (native prompt), then re-run with the recorded
# control signals injected in a permuted temporal order (chunks SPLIT..N-1
# first, then 0..SPLIT-1). Tests whether the 5-D bottleneck encodes reusable
# per-chunk control or just per-sample task identity.
set -eo pipefail

REPO=/scratch/gpfs/EYSENBACH/ss5822/vla-moe-diversity
CKPT=${REPO}/outputs/pi0_moe_bottleneck5/checkpoints/010000/pretrained_model
BASE_OUT=${REPO}/outputs/eval/bottleneck5_step010k_replay
OUT_RECORD=${BASE_OUT}/record
OUT_REPLAY=${BASE_OUT}/replay
LOG_DIR=${REPO}/logs
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/eval_bottleneck5_010k_replay_$(date +%Y%m%d_%H%M%S).log"

export HF_LEROBOT_HOME=/scratch/gpfs/EYSENBACH/ss5822/data
export HF_HOME=/scratch/gpfs/EYSENBACH/ss5822/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export REPLAY_PERM_SPLIT=4

cd "${REPO}"
rm -rf "${BASE_OUT}"

source /usr/share/Modules/init/bash
module load anaconda3/2025.6
source activate lerobot3

ROBOSUITE_MACROS_PRIVATE=$(python -c "import robosuite, os; print(os.path.join(os.path.dirname(robosuite.__file__), 'macros_private.py'))" 2>/dev/null || true)
if [ -n "${ROBOSUITE_MACROS_PRIVATE}" ] && [ ! -f "${ROBOSUITE_MACROS_PRIVATE}" ]; then
  echo "FILE_LOGGING_LEVEL = None" > "${ROBOSUITE_MACROS_PRIVATE}"
fi

run_phase () {
  local MODE=$1
  local OUT=$2
  echo "=========================================================="
  echo "[replay-eval] phase=${MODE}  out=${OUT}"
  echo "=========================================================="
  REPLAY_MODE=${MODE} python -m lerobot.scripts.replay_z_eval \
    --policy.path="${CKPT}" \
    --policy.device=cuda \
    --env.type=libero \
    --env.task=libero_10 \
    --env.task_ids='[0]' \
    --eval.batch_size=1 \
    --eval.n_episodes=1 \
    --output_dir="${OUT}" \
    --job_name=eval_bottleneck5_010k_replay_${MODE} \
    --seed=1000 \
    '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
}

{
  run_phase record "${OUT_RECORD}"
  run_phase replay "${OUT_REPLAY}"
} 2>&1 | tee "${LOG}"
