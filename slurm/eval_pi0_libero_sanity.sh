#!/bin/bash
# Sanity check: eval lerobot/pi0_libero (Pi0 already finetuned on Libero by
# Physical Intelligence) on libero_10 using the same eval harness we use for
# our MoE/bottleneck experiments. If this gets near published numbers
# (~85-95% on libero_10), our eval pipeline is correct and the 0% / 10%
# we see on our finetunes is purely a training-pipeline issue. If this
# ALSO bottoms out, there's a bug in eval/preprocessing.
set -eo pipefail

REPO=/scratch/gpfs/EYSENBACH/ss5822/vla-moe-diversity
OUT=${REPO}/outputs/eval/pi0_libero_sanity
LOG_DIR=${REPO}/logs
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/eval_pi0_libero_sanity_$(date +%Y%m%d_%H%M%S).log"

# Allow online HF only for this download
export HF_LEROBOT_HOME=/scratch/gpfs/EYSENBACH/ss5822/data
export HF_HOME=/scratch/gpfs/EYSENBACH/ss5822/cache/huggingface
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
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

echo "[pi0_libero_sanity $(date +%H:%M:%S)] downloading + evaluating lerobot/pi0_libero" | tee -a "${LOG}"

# n_action_steps=10 to match openpi published protocol; control_mode=relative
# (default; pi0_libero was trained on libero deltas, same as our v5).
python -m lerobot.scripts.lerobot_eval \
  --policy.path=lerobot/pi0_libero \
  --policy.device=cuda \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task=libero_10 \
  --env.task_ids='[0,1,2,3,4,5,6,7,8,9]' \
  --eval.batch_size=1 \
  --eval.n_episodes=5 \
  --output_dir="${OUT}" \
  --job_name=eval_pi0_libero_sanity \
  --seed=1000 \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.wrist_image": "observation.images.camera2"}' \
  2>&1 | tee -a "${LOG}"

PC=$(python - <<PY 2>/dev/null
import json
d=json.load(open("${OUT}/eval_info.json"))
print(d['overall']['pc_success'])
PY
)
echo "[pi0_libero_sanity $(date +%H:%M:%S)] pc_success=${PC}" | tee -a "${LOG}"
