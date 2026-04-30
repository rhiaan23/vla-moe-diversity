#!/bin/bash
# Local v5 eval on LIBERO-10 with router-decision capture.
# Output: <OUTDIR>/{videos/, eval_info.json, routing_log.json, expert_freq.{csv,png}}
set -uo pipefail

REPO=/scratch/gpfs/FHEIDE/rj2807/vla-moe-diversity
LOGDIR=/scratch/gpfs/FHEIDE/rj2807/logs
CKPT=/scratch/gpfs/FHEIDE/rj2807/outputs/pi0_moe_whole_v5_h100/checkpoints/last/pretrained_model
OUTDIR=/scratch/gpfs/FHEIDE/rj2807/outputs/evals/v5_libero_10
mkdir -p "$LOGDIR"

export HF_LEROBOT_HOME=/scratch/gpfs/FHEIDE/rj2807/lerobot_data
export HF_HOME=/scratch/gpfs/FHEIDE/rj2807/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0

source "$REPO/.venv/bin/activate"

# Robosuite expects this file; create empty if missing.
ROBOSUITE_MACROS_PRIVATE=$(python -c "import robosuite, os; print(os.path.join(os.path.dirname(robosuite.__file__), 'macros_private.py'))" 2>/dev/null || true)
if [ -n "$ROBOSUITE_MACROS_PRIVATE" ] && [ ! -f "$ROBOSUITE_MACROS_PRIVATE" ]; then
  echo "FILE_LOGGING_LEVEL = None" > "$ROBOSUITE_MACROS_PRIVATE"
fi

rm -rf "$OUTDIR"

python -m lerobot.scripts.eval_v5_routing \
  --policy.path="$CKPT" \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.batch_size=5 \
  --eval.n_episodes=20 \
  --policy.device=cuda \
  --output_dir="$OUTDIR" \
  --job_name=eval_v5_libero_10 \
  --seed=1000 \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
