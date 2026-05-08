#!/bin/bash
# Local v5 routing eval on LIBERO-10 tasks 2/4/5, 2 episodes each.
# Output: outputs/eval/pi0_moe_whole_v5_h100_step020000_tasks245/{videos/, eval_info.json, routing_log.json, expert_freq.{csv,png}}
set -o pipefail

REPO=/scratch/gpfs/EYSENBACH/ss5822/vla-moe-diversity
CKPT=${REPO}/checkpoints/pi0_moe_whole_v5_h100/checkpoints/020000/pretrained_model
OUTDIR=${REPO}/outputs/eval/pi0_moe_whole_v5_h100_step020000_tasks245

source /usr/share/Modules/init/bash
module load anaconda3/2024.2
source activate lerobot3

export HF_LEROBOT_HOME=/scratch/gpfs/EYSENBACH/ss5822/data
export HF_HOME=${HOME}/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0

ROBOSUITE_MACROS_PRIVATE=$(python -c "import robosuite, os; print(os.path.join(os.path.dirname(robosuite.__file__), 'macros_private.py'))" 2>/dev/null || true)
if [ -n "${ROBOSUITE_MACROS_PRIVATE}" ] && [ ! -f "${ROBOSUITE_MACROS_PRIVATE}" ]; then
  echo "FILE_LOGGING_LEVEL = None" > "${ROBOSUITE_MACROS_PRIVATE}"
fi

rm -rf "${OUTDIR}"

cd "${REPO}"

exec python -m lerobot.scripts.eval_v5_routing \
  --policy.path="${CKPT}" \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_10 \
  --env.task_ids='[2,4,5]' \
  --eval.batch_size=3 \
  --eval.n_episodes=3 \
  --output_dir="${OUTDIR}" \
  --job_name=eval_v5_tasks245 \
  --seed=1000 \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
