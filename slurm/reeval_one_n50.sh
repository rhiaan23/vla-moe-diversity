#!/bin/bash
# Same as reeval_one.sh but evaluates at the saved n_action_steps (50 for Pi0
# checkpoints) instead of the openpi-published 10. Used to ablate the
# n_action_steps lever at corrected camera rename.
# Usage: reeval_one_n50.sh <TAG> <CKPT_DIR>
set -eo pipefail

TAG=${1:?"Usage: $0 <TAG> <CKPT_DIR>"}
CKPT=${2:?"Usage: $0 <TAG> <CKPT_DIR>"}

REPO=/scratch/gpfs/EYSENBACH/ss5822/vla-moe-diversity
OUT=${REPO}/outputs/eval/${TAG}_n50
LOG_DIR=${REPO}/logs
mkdir -p "${LOG_DIR}"
LOG=${LOG_DIR}/eval_${TAG}_n50_$(date +%Y%m%d_%H%M%S).log

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
[ -n "${ROBOSUITE_MACROS_PRIVATE}" ] && [ ! -f "${ROBOSUITE_MACROS_PRIVATE}" ] && \
  echo "FILE_LOGGING_LEVEL = None" > "${ROBOSUITE_MACROS_PRIVATE}"

echo "[reeval_one_n50 $(date +%H:%M:%S)] tag=${TAG}  ckpt=${CKPT}  out=${OUT}  host=$(hostname)" | tee -a "${LOG}"

# NB: NO --policy.n_action_steps override here; the saved config (50) is used.
python -m lerobot.scripts.eval_v5_routing \
  --policy.path="${CKPT}" \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_10 \
  --env.task_ids='[0,1,2,3,4,5,6,7,8,9]' \
  --eval.batch_size=1 \
  --eval.n_episodes=5 \
  --output_dir="${OUT}" \
  --job_name=eval_${TAG}_n50 \
  --seed=1000 \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  2>&1 | tee -a "${LOG}"

python "${REPO}/scripts/annotate_v5_videos.py" \
  --eval-dir "${OUT}" \
  --suite libero_10 \
  --batch-size 1 \
  --n-episodes 5 \
  --max-episodes 5 \
  --chunk 50 \
  2>&1 | tee -a "${LOG}"

PC=$(python - <<PY 2>/dev/null
import json
d = json.load(open("${OUT}/eval_info.json"))
print(d['overall']['pc_success'])
PY
)
echo "[reeval_one_n50 $(date +%H:%M:%S)] DONE tag=${TAG}  host=$(hostname)  pc_success=${PC}" | tee -a "${LOG}"
