#!/bin/bash
# Same as eval_pi0_libero_sanity.sh but WITHOUT --rename_map. pi0_libero's
# expected input keys (per its config.json) are observation.images.image,
# observation.images.image2, observation.state — which is exactly what the
# libero env emits natively. The rename map we used for our v5 / MoE
# models ('image -> camera1', 'wrist_image -> camera2') was both renaming
# pi0_libero's expected key away AND failing to match the env's actual
# wrist key (image2, not wrist_image), so camera2 was zero-filled.
set -eo pipefail

REPO=/scratch/gpfs/EYSENBACH/ss5822/vla-moe-diversity
OUT=${REPO}/outputs/eval/pi0_libero_norename
LOG_DIR=${REPO}/logs
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/eval_pi0_libero_norename_$(date +%Y%m%d_%H%M%S).log"

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

echo "[pi0_libero_norename $(date +%H:%M:%S)] eval pi0_libero with NO rename_map" | tee -a "${LOG}"

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
  --job_name=eval_pi0_libero_norename \
  --seed=1000 \
  2>&1 | tee -a "${LOG}"

PC=$(python - <<PY 2>/dev/null
import json
d=json.load(open("${OUT}/eval_info.json"))
print(d['overall']['pc_success'])
PY
)
echo "[pi0_libero_norename $(date +%H:%M:%S)] pc_success=${PC}" | tee -a "${LOG}"
