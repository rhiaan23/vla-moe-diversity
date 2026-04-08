#!/bin/bash
#SBATCH --job-name=eval-gen-ms-obj
#SBATCH --account=eladgroup
#SBATCH --partition=pli
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/gpfs/EHAZAN/tharuntk/logs/eval_gen_moe_standard_object_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/tharuntk/logs/eval_gen_moe_standard_object_%j.err

mkdir -p /scratch/gpfs/EHAZAN/tharuntk/logs

export HF_LEROBOT_HOME=/scratch/gpfs/EHAZAN/tharuntk/lerobot_data
export HF_HOME=/scratch/gpfs/EHAZAN/tharuntk/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl

source /home/tt6444/lerobot/.venv/bin/activate

ROBOSUITE_MACROS_PRIVATE=$(python -c "import robosuite; import os; print(os.path.join(os.path.dirname(robosuite.__file__), 'macros_private.py'))" 2>/dev/null || true)
if [ -n "$ROBOSUITE_MACROS_PRIVATE" ] && [ ! -f "$ROBOSUITE_MACROS_PRIVATE" ]; then
  echo "FILE_LOGGING_LEVEL = None" > "$ROBOSUITE_MACROS_PRIVATE"
fi

python -m lerobot.scripts.lerobot_eval \
  --policy.path=/scratch/gpfs/EHAZAN/tharuntk/outputs/moe_standard/checkpoints/last/pretrained_model \
  --env.type=libero \
  --env.task=libero_object \
  --eval.batch_size=10 \
  --eval.n_episodes=50 \
  --policy.device=cuda \
  --output_dir=/scratch/gpfs/EHAZAN/tharuntk/outputs/evals/generalization/moe_standard/libero_object \
  --job_name=eval_gen_moe_standard_object \
  --seed=1000 \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
