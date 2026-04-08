#!/bin/bash
#SBATCH --job-name=smolvla-baseline
#SBATCH --account=eladgroup
#SBATCH --partition=pli
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gpfs/EHAZAN/tharuntk/logs/baseline_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/tharuntk/logs/baseline_%j.err

mkdir -p /scratch/gpfs/EHAZAN/tharuntk/logs

export HF_LEROBOT_HOME=/scratch/gpfs/EHAZAN/tharuntk/lerobot_data
export HF_HOME=/scratch/gpfs/EHAZAN/tharuntk/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source /home/tt6444/lerobot/.venv/bin/activate

rm -rf /scratch/gpfs/EHAZAN/tharuntk/outputs/baseline

python -m lerobot.scripts.lerobot_train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/libero_10 \
  --batch_size=256 \
  --num_workers=7 \
  --steps=50000 \
  --output_dir=/scratch/gpfs/EHAZAN/tharuntk/outputs/baseline \
  --wandb.enable=true \
  --wandb.mode=offline \
  --wandb.project=vla-moe-diversity \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.wrist_image": "observation.images.camera2"}'
