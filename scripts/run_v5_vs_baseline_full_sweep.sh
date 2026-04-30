#!/bin/bash
# Sequential 10-eval sweep: v5 vs dense-Pi0 baseline
#
# Goal 1 (multi-seed in-distribution): both models on libero_10 at seeds 2000, 3000.
#   Combined with the existing seed=1000 evals, gives 3-seed paired comparison.
# Goal 2 (held-out generalization): both models on libero_{goal,object,spatial} at seed=1000.
#
# Total: 10 fresh evals × ~30 min = ~5 h on H100.
# Each eval writes to outputs/evals/sweep/<model>_<task>_seed<seed>/
#
# Run with:  setsid nohup bash scripts/run_v5_vs_baseline_full_sweep.sh > /scratch/gpfs/FHEIDE/rj2807/logs/sweep.log 2>&1 < /dev/null &

set -uo pipefail

REPO=/scratch/gpfs/FHEIDE/rj2807/vla-moe-diversity
SWEEPDIR=/scratch/gpfs/FHEIDE/rj2807/outputs/evals/sweep
mkdir -p "$SWEEPDIR"

cd "$REPO"
source .venv/bin/activate

export HF_LEROBOT_HOME=/scratch/gpfs/FHEIDE/rj2807/lerobot_data
export HF_HOME=/scratch/gpfs/FHEIDE/rj2807/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0

ROBOSUITE_MACROS_PRIVATE=$(python -c "import robosuite, os; print(os.path.join(os.path.dirname(robosuite.__file__), 'macros_private.py'))" 2>/dev/null || true)
if [ -n "$ROBOSUITE_MACROS_PRIVATE" ] && [ ! -f "$ROBOSUITE_MACROS_PRIVATE" ]; then
  echo "FILE_LOGGING_LEVEL = None" > "$ROBOSUITE_MACROS_PRIVATE"
fi

CKPT_BL=/scratch/gpfs/FHEIDE/rj2807/outputs/pi0_baseline_v2/checkpoints/last/pretrained_model
CKPT_V5=/scratch/gpfs/FHEIDE/rj2807/outputs/pi0_moe_whole_v5_h100/checkpoints/last/pretrained_model

run_eval() {
  local model="$1" ckpt="$2" task="$3" seed="$4"
  local out="$SWEEPDIR/${model}_${task}_seed${seed}"
  echo "============================================================"
  echo "[$(date '+%H:%M:%S')] eval: model=$model  task=$task  seed=$seed"
  echo "  out=$out"
  echo "============================================================"
  rm -rf "$out"
  python -m lerobot.scripts.lerobot_eval \
    --policy.path="$ckpt" \
    --env.type=libero \
    --env.task="$task" \
    --eval.batch_size=5 \
    --eval.n_episodes=20 \
    --policy.device=cuda \
    --output_dir="$out" \
    --job_name="sweep_${model}_${task}_seed${seed}" \
    --seed="$seed" \
    '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date '+%H:%M:%S')] FAILED rc=$rc for model=$model task=$task seed=$seed"
  else
    # Print success rate inline
    python -c "
import json
d = json.load(open('$out/eval_info.json'))
per = [sum(t['metrics']['successes'])/len(t['metrics']['successes']) for t in d['per_task']]
print(f'  → mean success: {sum(per)/len(per):.3f}  per-task: {[round(x,2) for x in per]}')
" 2>/dev/null || true
  fi
}

# --- Goal 1: multi-seed libero_10 (existing seed=1000 not re-run) ---
for SEED in 2000 3000; do
  run_eval baseline "$CKPT_BL" libero_10 "$SEED"
  run_eval v5       "$CKPT_V5" libero_10 "$SEED"
done

# --- Goal 2: held-out suites at seed=1000 ---
for TASK in libero_goal libero_object libero_spatial; do
  run_eval baseline "$CKPT_BL" "$TASK" 1000
  run_eval v5       "$CKPT_V5" "$TASK" 1000
done

echo "============================================================"
echo "[$(date '+%H:%M:%S')] sweep done"
echo "============================================================"
ls -la "$SWEEPDIR"
