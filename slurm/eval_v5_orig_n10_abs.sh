#!/bin/bash
# Re-eval the original v5 checkpoint with n_action_steps=10 AND
# env.control_mode=absolute. v5 was trained with use_relative_actions=False
# so the policy outputs absolute target poses; the libero env default is
# 'relative' which would interpret those as deltas. This run pairs the eval
# control mode with the policy's training mode.
set -eo pipefail

REPO=/scratch/gpfs/EYSENBACH/ss5822/vla-moe-diversity
CKPT=/scratch/gpfs/EYSENBACH/cn7658/VLA/compositionality/vla-moe-diversity/outputs/hf_cache/pi0_moe_whole_v5_h100/checkpoints/020000/pretrained_model
OUT=${REPO}/outputs/eval/v5_orig_n10_abs
LOG_DIR=${REPO}/logs
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/eval_v5_orig_n10_abs_$(date +%Y%m%d_%H%M%S).log"

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

ROBOSUITE_MACROS_PRIVATE=$(python -c "import robosuite, os; print(os.path.join(os.path.dirname(robosuite.__file__), 'macros_private.py'))" 2>/dev/null || true)
if [ -n "${ROBOSUITE_MACROS_PRIVATE}" ] && [ ! -f "${ROBOSUITE_MACROS_PRIVATE}" ]; then
  echo "FILE_LOGGING_LEVEL = None" > "${ROBOSUITE_MACROS_PRIVATE}"
fi

echo "[eval-v5-orig-n10-abs $(date +%H:%M:%S)] ckpt=${CKPT}  out=${OUT}" | tee -a "${LOG}"

python -m lerobot.scripts.eval_v5_routing \
  --policy.path="${CKPT}" \
  --policy.device=cuda \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task=libero_10 \
  --env.task_ids='[0,1,2,3,4,5,6,7,8,9]' \
  --env.control_mode=absolute \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --output_dir="${OUT}" \
  --job_name=eval_v5_orig_n10_abs \
  --seed=1000 \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.wrist_image": "observation.images.camera2"}' \
  2>&1 | tee -a "${LOG}"

echo "[eval-v5-orig-n10-abs $(date +%H:%M:%S)] running annotation with chunk=10" | tee -a "${LOG}"
python "${REPO}/scripts/annotate_v5_videos.py" \
  --eval-dir "${OUT}" \
  --suite libero_10 \
  --batch-size 1 \
  --n-episodes 1 \
  --max-episodes 1 \
  --chunk 10 \
  2>&1 | tee -a "${LOG}"

PC=$(python - <<PY 2>/dev/null
import json
d=json.load(open("${OUT}/eval_info.json"))
print(d['overall']['pc_success'])
PY
)
echo "[eval-v5-orig-n10-abs $(date +%H:%M:%S)] pc_success=${PC}" | tee -a "${LOG}"
