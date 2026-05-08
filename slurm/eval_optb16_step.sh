#!/bin/bash
# Run v5-routing eval + annotation for a single pi0_bottleneck16_optb checkpoint.
# Usage: eval_optb16_step.sh <STEP>     e.g. eval_optb16_step.sh 2000
set -eo pipefail

STEP=${1:?"Usage: $0 <STEP>"}

REPO=/scratch/gpfs/EYSENBACH/ss5822/vla-moe-diversity
STEP_PADDED=$(printf "%06d" "${STEP}")
STEP_K=$(printf "%03dk" $((STEP/1000)))
CKPT="${REPO}/outputs/pi0_bottleneck16_optb/checkpoints/${STEP_PADDED}/pretrained_model"
OUT="${REPO}/outputs/eval/optb16_step${STEP_K}_v5"
LOG_DIR=${REPO}/logs
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/eval_optb16_${STEP_K}_v5_$(date +%Y%m%d_%H%M%S).log"

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

echo "[eval-optb16 $(date +%H:%M:%S)] step=${STEP}  ckpt=${CKPT}  out=${OUT}" | tee -a "${LOG}"

python -m lerobot.scripts.eval_v5_routing \
  --policy.path="${CKPT}" \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_10 \
  --env.task_ids='[0,1,2,3,4,5,6,7,8,9]' \
  --policy.n_action_steps=10 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --output_dir="${OUT}" \
  --job_name=eval_optb16_${STEP_K}_v5 \
  --seed=1000 \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  2>&1 | tee -a "${LOG}"

echo "[eval-optb16 $(date +%H:%M:%S)] running annotation" | tee -a "${LOG}"
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
echo "[eval-optb16 $(date +%H:%M:%S)] step=${STEP} pc_success=${PC}" | tee -a "${LOG}"
