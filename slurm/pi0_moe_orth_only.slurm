#!/bin/bash
#SBATCH --job-name=pi0-moe-orth-only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gpfs/FHEIDE/rj2807/logs/pi0_moe_orth_only_%j.out
#SBATCH --error=/scratch/gpfs/FHEIDE/rj2807/logs/pi0_moe_orth_only_%j.err

mkdir -p /scratch/gpfs/FHEIDE/rj2807/logs

export HF_LEROBOT_HOME=/scratch/gpfs/FHEIDE/rj2807/lerobot_data
export HF_HOME=/scratch/gpfs/FHEIDE/rj2807/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source /scratch/gpfs/FHEIDE/rj2807/vla-moe-diversity/.venv/bin/activate

rm -rf /scratch/gpfs/FHEIDE/rj2807/outputs/pi0_moe_orth_only

python -m lerobot.scripts.lerobot_train \
  --policy.path=lerobot/pi0 \
  --policy.push_to_hub=false \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.use_moe=true \
  --policy.use_diversity_loss=true \
  --policy.moe_lambda_orth=0.05 \
  --policy.moe_lambda_disc=0.0 \
  --dataset.repo_id=lerobot/libero_10 \
  --batch_size=64 \
  --num_workers=4 \
  --steps=50000 \
  --output_dir=/scratch/gpfs/FHEIDE/rj2807/outputs/pi0_moe_orth_only \
  --wandb.enable=true \
  --wandb.mode=offline \
  --wandb.project=vla-moe-diversity \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.wrist_image": "observation.images.camera2"}'
