#!/bin/bash
# LIBERO-Spatial generalization eval for pi0_moe_upcycled_v3_h100 on fisec RTX6000.
# ADJUST the SBATCH header below to match your fisec cluster's conventions.
#SBATCH --job-name=eval-gen-v3-spatial
#SBATCH --account=fheide
#SBATCH --partition=fisec
#SBATCH --constraint=rtx_6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/gpfs/FHEIDE/rj2807/logs/eval_gen_v3_spatial_%j.out
#SBATCH --error=/scratch/gpfs/FHEIDE/rj2807/logs/eval_gen_v3_spatial_%j.err

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
  --policy.path=/scratch/gpfs/FHEIDE/rj2807/outputs/pi0_moe_upcycled_v3_h100/checkpoints/last/pretrained_model \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.batch_size=10 \
  --eval.n_episodes=50 \
  --policy.device=cuda \
  --output_dir=/scratch/gpfs/FHEIDE/rj2807/outputs/evals_v3/moe_upcycled_v3/libero_spatial \
  --job_name=eval_gen_v3_spatial \
  --seed=1000 \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
