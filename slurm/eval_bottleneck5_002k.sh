#!/bin/bash
# One-shot eval of the step-2K checkpoint from the bottleneck5 training run.
# Runs 1 rollout per LIBERO-10 task on a non-training GPU.
set -eo pipefail

REPO=/scratch/gpfs/EYSENBACH/ss5822/vla-moe-diversity
CKPT=${REPO}/outputs/pi0_moe_bottleneck5/checkpoints/002000/pretrained_model
OUT=${REPO}/outputs/eval/bottleneck5_step002k
LOG_DIR=${REPO}/logs
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/eval_bottleneck5_002k_$(date +%Y%m%d_%H%M%S).log"

export HF_LEROBOT_HOME=/scratch/gpfs/EYSENBACH/ss5822/data
export HF_HOME=/scratch/gpfs/EYSENBACH/ss5822/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${REPO}"
rm -rf "${OUT}"

source /usr/share/Modules/init/bash
module load anaconda3/2025.6
source activate lerobot3

# robosuite likes a private macros file in its install dir.
ROBOSUITE_MACROS_PRIVATE=$(python -c "import robosuite; import os; print(os.path.join(os.path.dirname(robosuite.__file__), 'macros_private.py'))" 2>/dev/null || true)
if [ -n "${ROBOSUITE_MACROS_PRIVATE}" ] && [ ! -f "${ROBOSUITE_MACROS_PRIVATE}" ]; then
  echo "FILE_LOGGING_LEVEL = None" > "${ROBOSUITE_MACROS_PRIVATE}"
fi

exec python -m lerobot.scripts.lerobot_eval \
  --policy.path="${CKPT}" \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_10 \
  --env.task_ids='[0,1,2,3,4,5,6,7,8,9]' \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --output_dir="${OUT}" \
  --job_name=eval_bottleneck5_002k \
  --seed=1000 \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  2>&1 | tee "${LOG}"
