#!/bin/bash
#SBATCH --job-name=eval-v2-baseline-goal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/gpfs/FHEIDE/rj2807/logs/eval_v2_baseline_goal_%j.out
#SBATCH --error=/scratch/gpfs/FHEIDE/rj2807/logs/eval_v2_baseline_goal_%j.err

mkdir -p /scratch/gpfs/FHEIDE/rj2807/logs

export HF_LEROBOT_HOME=/scratch/gpfs/FHEIDE/rj2807/lerobot_data
export HF_HOME=/scratch/gpfs/FHEIDE/rj2807/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl

source /scratch/gpfs/FHEIDE/rj2807/vla-moe-diversity/.venv/bin/activate

ROBOSUITE_MACROS_PRIVATE=$(python -c "import robosuite; import os; print(os.path.join(os.path.dirname(robosuite.__file__), 'macros_private.py'))" 2>/dev/null || true)
if [ -n "$ROBOSUITE_MACROS_PRIVATE" ] && [ ! -f "$ROBOSUITE_MACROS_PRIVATE" ]; then
  echo "FILE_LOGGING_LEVEL = None" > "$ROBOSUITE_MACROS_PRIVATE"
fi

python -m lerobot.scripts.lerobot_eval \
  --policy.path=/scratch/gpfs/FHEIDE/rj2807/outputs/pi0_baseline_v2/checkpoints/020000/pretrained_model \
  --env.type=libero \
  --env.task=libero_goal \
  --eval.batch_size=5 \
  --eval.n_episodes=50 \
  --policy.device=cuda \
  --output_dir=/scratch/gpfs/FHEIDE/rj2807/outputs/evals_v2/baseline/libero_goal \
  --job_name=eval_v2_baseline_goal \
  --seed=1000 \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
