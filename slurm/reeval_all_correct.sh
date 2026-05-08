#!/bin/bash
# Re-eval the four most-recent checkpoints with the CORRECT eval rename map
# (observation.images.image2 -> camera2; the env emits image2, NOT
# wrist_image — every prior MoE/bottleneck eval was zero-filling camera2).
# n_action_steps=10 to match openpi's published Pi0 LIBERO inference protocol.
# 5 episodes per task × 10 tasks = 50 episodes per checkpoint.
set -eo pipefail

REPO=/scratch/gpfs/EYSENBACH/ss5822/vla-moe-diversity
LOG_DIR=${REPO}/logs
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/reeval_all_correct_$(date +%Y%m%d_%H%M%S).log"

export HF_LEROBOT_HOME=/scratch/gpfs/EYSENBACH/ss5822/data
export HF_HOME=/scratch/gpfs/EYSENBACH/ss5822/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${REPO}"

source /usr/share/Modules/init/bash
module load anaconda3/2025.6
source activate lerobot3

ROBOSUITE_MACROS_PRIVATE=$(python -c "import robosuite, os; print(os.path.join(os.path.dirname(robosuite.__file__), 'macros_private.py'))" 2>/dev/null || true)
[ -n "${ROBOSUITE_MACROS_PRIVATE}" ] && [ ! -f "${ROBOSUITE_MACROS_PRIVATE}" ] && \
  echo "FILE_LOGGING_LEVEL = None" > "${ROBOSUITE_MACROS_PRIVATE}"

run_one () {
  local TAG=$1
  local CKPT=$2
  local OUT=${REPO}/outputs/eval/${TAG}_correct
  local SUBLOG=${LOG_DIR}/eval_${TAG}_correct_$(date +%Y%m%d_%H%M%S).log

  echo "" | tee -a "${LOG}"
  echo "============================================================" | tee -a "${LOG}"
  echo "[reeval $(date +%H:%M:%S)] tag=${TAG}" | tee -a "${LOG}"
  echo "  ckpt=${CKPT}" | tee -a "${LOG}"
  echo "  out=${OUT}" | tee -a "${LOG}"
  echo "============================================================" | tee -a "${LOG}"

  rm -rf "${OUT}"

  python -m lerobot.scripts.eval_v5_routing \
    --policy.path="${CKPT}" \
    --policy.device=cuda \
    --policy.n_action_steps=10 \
    --env.type=libero \
    --env.task=libero_10 \
    --env.task_ids='[0,1,2,3,4,5,6,7,8,9]' \
    --eval.batch_size=1 \
    --eval.n_episodes=5 \
    --output_dir="${OUT}" \
    --job_name=eval_${TAG}_correct \
    --seed=1000 \
    '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
    2>&1 | tee "${SUBLOG}" | tail -1
  EVAL_RC=${PIPESTATUS[0]}

  if [ "${EVAL_RC}" -ne 0 ]; then
    echo "[reeval $(date +%H:%M:%S)] FAILED tag=${TAG} (rc=${EVAL_RC}); sublog=${SUBLOG}" | tee -a "${LOG}"
    return
  fi

  python "${REPO}/scripts/annotate_v5_videos.py" \
    --eval-dir "${OUT}" \
    --suite libero_10 \
    --batch-size 1 \
    --n-episodes 5 \
    --max-episodes 5 \
    --chunk 10 \
    2>&1 >> "${SUBLOG}"

  PC=$(python - <<PY 2>/dev/null
import json
d = json.load(open("${OUT}/eval_info.json"))
print(d['overall']['pc_success'])
PY
)
  echo "[reeval $(date +%H:%M:%S)] DONE tag=${TAG}  pc_success=${PC}" | tee -a "${LOG}"
}

run_one bottleneck5_nonoptb         "${REPO}/outputs/pi0_moe_bottleneck5/checkpoints/last/pretrained_model"
run_one bottleneck5_optb            "${REPO}/outputs/pi0_bottleneck5_optb/checkpoints/last/pretrained_model"
# bottleneck16_optb and bottleneck16_optb_libero40 dispatched to other hosts in parallel — see slurm/reeval_one.sh
# run_one bottleneck16_optb           "${REPO}/outputs/pi0_bottleneck16_optb/checkpoints/last/pretrained_model"
# run_one bottleneck16_optb_libero40  "${REPO}/outputs/pi0_bottleneck16_optb_libero40/checkpoints/last/pretrained_model"

echo "" | tee -a "${LOG}"
echo "=== SUMMARY ===" | tee -a "${LOG}"
for t in bottleneck5_nonoptb bottleneck5_optb bottleneck16_optb bottleneck16_optb_libero40; do
  f=${REPO}/outputs/eval/${t}_correct/eval_info.json
  if [ -f "$f" ]; then
    PC=$(python -c "import json;d=json.load(open('$f'))['overall'];print(d['pc_success'])" 2>/dev/null)
    echo "  ${t}: pc_success=${PC}" | tee -a "${LOG}"
  else
    echo "  ${t}: missing" | tee -a "${LOG}"
  fi
done
