#!/bin/bash
# Re-eval original v5 with the CORRECT rename map. The env emits
# observation.images.image2 (not wrist_image), so camera2 must be sourced
# from image2 — every prior MoE/bottleneck eval mistakenly mapped from
# wrist_image (a dataset key, not an env key) and ran with camera2 zero-filled.
set -eo pipefail
REPO=/scratch/gpfs/EYSENBACH/ss5822/vla-moe-diversity
CKPT=/scratch/gpfs/EYSENBACH/cn7658/VLA/compositionality/vla-moe-diversity/outputs/hf_cache/pi0_moe_whole_v5_h100/checkpoints/020000/pretrained_model
OUT=${REPO}/outputs/eval/v5_orig_correct
LOG=${REPO}/logs/eval_v5_orig_correct_$(date +%Y%m%d_%H%M%S).log
mkdir -p "${REPO}/logs"
export HF_LEROBOT_HOME=/scratch/gpfs/EYSENBACH/ss5822/data
export HF_HOME=/scratch/gpfs/EYSENBACH/ss5822/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO}"; rm -rf "${OUT}"
source /usr/share/Modules/init/bash; module load anaconda3/2025.6; source activate lerobot3
ROBOSUITE_MACROS_PRIVATE=$(python -c "import robosuite, os; print(os.path.join(os.path.dirname(robosuite.__file__), 'macros_private.py'))" 2>/dev/null || true)
[ -n "${ROBOSUITE_MACROS_PRIVATE}" ] && [ ! -f "${ROBOSUITE_MACROS_PRIVATE}" ] && echo "FILE_LOGGING_LEVEL = None" > "${ROBOSUITE_MACROS_PRIVATE}"

python -m lerobot.scripts.eval_v5_routing \
  --policy.path="${CKPT}" --policy.device=cuda --policy.n_action_steps=10 \
  --env.type=libero --env.task=libero_10 --env.task_ids='[0,1,2,3,4,5,6,7,8,9]' \
  --eval.batch_size=1 --eval.n_episodes=5 \
  --output_dir="${OUT}" --job_name=eval_v5_orig_correct --seed=1000 \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  2>&1 | tee "${LOG}"

python "${REPO}/scripts/annotate_v5_videos.py" --eval-dir "${OUT}" --suite libero_10 --batch-size 1 --n-episodes 5 --max-episodes 5 --chunk 10 2>&1 | tee -a "${LOG}"

python -c "import json;d=json.load(open('${OUT}/eval_info.json'))['overall'];print(f'pc_success={d[\"pc_success\"]}')" | tee -a "${LOG}"
