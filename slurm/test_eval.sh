#!/bin/bash
# Quick smoke test: run 1 episode of each model type to check for OOM
set -e

export HF_LEROBOT_HOME=/scratch/gpfs/EHAZAN/tharuntk/lerobot_data
export HF_HOME=/scratch/gpfs/EHAZAN/tharuntk/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl

source /home/tt6444/lerobot/.venv/bin/activate

# Ensure robosuite doesn't fail on log file
ROBOSUITE_MACROS_PRIVATE=$(python -c "import robosuite; import os; print(os.path.join(os.path.dirname(robosuite.__file__), 'macros_private.py'))" 2>/dev/null || true)
if [ -n "$ROBOSUITE_MACROS_PRIVATE" ] && [ ! -f "$ROBOSUITE_MACROS_PRIVATE" ]; then
  echo "FILE_LOGGING_LEVEL = None" > "$ROBOSUITE_MACROS_PRIVATE"
fi

RENAME='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'

for MODEL in baseline moe_standard moe_diversity; do
  CKPT=/scratch/gpfs/EHAZAN/tharuntk/outputs/${MODEL}/checkpoints/last/pretrained_model
  OUT=/tmp/eval_smoke_test/${MODEL}
  rm -rf "$OUT"

  echo ""
  echo "=========================================="
  echo "SMOKE TEST: ${MODEL} on libero_10 (batch_size=10, 1 episode)"
  echo "=========================================="

  python -m lerobot.scripts.lerobot_eval \
    --policy.path="${CKPT}" \
    --env.type=libero \
    --env.task=libero_10 \
    --eval.batch_size=1 \
    --eval.n_episodes=1 \
    --policy.device=cuda \
    --output_dir="${OUT}" \
    --job_name="smoke_test_${MODEL}" \
    --seed=1000 \
    "--rename_map=${RENAME}"

  echo ">>> ${MODEL}: SUCCESS"
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
  echo ""
done

echo "===== ALL SMOKE TESTS PASSED ====="
