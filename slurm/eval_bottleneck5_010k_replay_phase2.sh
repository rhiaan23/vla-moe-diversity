#!/bin/bash
# Phase 2: replay using stored per-chunk (z, idx, wts) from phase 1, with
# chunks reordered as [SPLIT..N-1, 0..SPLIT-1]. Default SPLIT=7 puts
# chunks 7-10 (tomato) before chunks 0-6 (alphabet) on task 0.
set -eo pipefail

REPO=/scratch/gpfs/EYSENBACH/ss5822/vla-moe-diversity
CKPT=${REPO}/outputs/pi0_moe_bottleneck5/checkpoints/010000/pretrained_model
BASE_OUT=${REPO}/outputs/eval/bottleneck5_step010k_replay
OUT_REPLAY=${BASE_OUT}/replay
LOG_DIR=${REPO}/logs
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/eval_bottleneck5_010k_replay_phase2_$(date +%Y%m%d_%H%M%S).log"

export HF_LEROBOT_HOME=/scratch/gpfs/EYSENBACH/ss5822/data
export HF_HOME=/scratch/gpfs/EYSENBACH/ss5822/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export REPLAY_PERM_SPLIT=${REPLAY_PERM_SPLIT:-7}

cd "${REPO}"
rm -rf "${OUT_REPLAY}"

source /usr/share/Modules/init/bash
module load anaconda3/2025.6
source activate lerobot3

ROBOSUITE_MACROS_PRIVATE=$(python -c "import robosuite, os; print(os.path.join(os.path.dirname(robosuite.__file__), 'macros_private.py'))" 2>/dev/null || true)
if [ -n "${ROBOSUITE_MACROS_PRIVATE}" ] && [ ! -f "${ROBOSUITE_MACROS_PRIVATE}" ]; then
  echo "FILE_LOGGING_LEVEL = None" > "${ROBOSUITE_MACROS_PRIVATE}"
fi

REPLAY_MODE=replay exec python -m lerobot.scripts.replay_z_eval \
  --policy.path="${CKPT}" \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_10 \
  --env.task_ids='[0]' \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --output_dir="${OUT_REPLAY}" \
  --job_name=eval_bottleneck5_010k_replay_replay \
  --seed=1000 \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  2>&1 | tee "${LOG}"
