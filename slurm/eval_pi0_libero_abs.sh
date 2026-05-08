#!/bin/bash
set -eo pipefail
REPO=/scratch/gpfs/EYSENBACH/ss5822/vla-moe-diversity
OUT=${REPO}/outputs/eval/pi0_libero_abs
LOG=${REPO}/logs/eval_pi0_libero_abs_$(date +%Y%m%d_%H%M%S).log
mkdir -p "${REPO}/logs"
export HF_LEROBOT_HOME=/scratch/gpfs/EYSENBACH/ss5822/data
export HF_HOME=/scratch/gpfs/EYSENBACH/ss5822/cache/huggingface
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO}"; rm -rf "${OUT}"
source /usr/share/Modules/init/bash; module load anaconda3/2025.6; source activate lerobot3
ROBOSUITE_MACROS_PRIVATE=$(python -c "import robosuite, os; print(os.path.join(os.path.dirname(robosuite.__file__), 'macros_private.py'))" 2>/dev/null || true)
[ -n "${ROBOSUITE_MACROS_PRIVATE}" ] && [ ! -f "${ROBOSUITE_MACROS_PRIVATE}" ] && echo "FILE_LOGGING_LEVEL = None" > "${ROBOSUITE_MACROS_PRIVATE}"

python -m lerobot.scripts.lerobot_eval \
  --policy.path=lerobot/pi0_libero --policy.device=cuda --policy.n_action_steps=10 \
  --env.type=libero --env.task=libero_10 --env.task_ids='[0,1,2,3,4,5,6,7,8,9]' \
  --env.control_mode=absolute --env.max_parallel_tasks=1 \
  --eval.batch_size=1 --eval.n_episodes=5 \
  --output_dir="${OUT}" --job_name=eval_pi0_libero_abs --seed=1000 \
  2>&1 | tee "${LOG}"

python -c "import json;d=json.load(open('${OUT}/eval_info.json'))['overall'];print(f'pc_success={d[\"pc_success\"]}')" | tee -a "${LOG}"
