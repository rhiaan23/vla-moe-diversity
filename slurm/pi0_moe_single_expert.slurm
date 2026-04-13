#!/bin/bash
#SBATCH --job-name=pi0-moe-single-expert
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/network/rj2807/logs/pi0_moe_single_expert_%j.out
#SBATCH --error=/scratch/network/rj2807/logs/pi0_moe_single_expert_%j.err

mkdir -p /scratch/network/rj2807/logs

export HF_LEROBOT_HOME=/scratch/network/rj2807/lerobot_data
export HF_HOME=/scratch/network/rj2807/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source /scratch/network/rj2807/vla-moe-diversity/.venv/bin/activate

rm -rf /scratch/network/rj2807/outputs/pi0_moe_single_expert

python -m lerobot.scripts.lerobot_train \
  --policy.path=lerobot/pi0 \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.push_to_hub=false \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.gradient_checkpointing=true \
  --policy.use_moe=true \
  --policy.use_diversity_loss=false \
  --policy.moe_num_experts=1 \
  --policy.moe_top_k=1 \
  --dataset.repo_id=lerobot/libero_10 \
  --batch_size=64 \
  --num_workers=4 \
  --steps=50000 \
  --output_dir=/scratch/network/rj2807/outputs/pi0_moe_single_expert \
  --wandb.enable=true \
  --wandb.mode=offline \
  --wandb.project=vla-moe-diversity \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.wrist_image": "observation.images.camera2"}'
