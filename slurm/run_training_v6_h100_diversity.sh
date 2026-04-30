#!/bin/bash
# v6 full-diversity run: post disc-loss-bugfix (moe.py 2026-04-28).
# Same hyperparams as v3_h100 (orth=0.05, disc=0.02) but disc loss now actually fires.
# Designed to run on della-stellato GPU 0 (H100 NVL 95GB) outside slurm, in a tmux session.
set -uo pipefail

REPO=/scratch/gpfs/FHEIDE/rj2807/vla-moe-diversity
LOGDIR=/scratch/gpfs/FHEIDE/rj2807/logs
OUTDIR=/scratch/gpfs/FHEIDE/rj2807/outputs/pi0_moe_v6_diversity_h100
mkdir -p "$LOGDIR"

export HF_LEROBOT_HOME=/scratch/gpfs/FHEIDE/rj2807/lerobot_data
export HF_HOME=/scratch/gpfs/FHEIDE/rj2807/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DIR=/scratch/gpfs/FHEIDE/rj2807/cache/wandb-home
export CUDA_VISIBLE_DEVICES=0

source "$REPO/.venv/bin/activate"

RESUME_FLAG=""
if [ "${RESUME:-0}" = "1" ]; then
  RESUME_FLAG="--resume=true"
else
  rm -rf "$OUTDIR"
fi

python -m lerobot.scripts.lerobot_train \
  $RESUME_FLAG \
  --policy.path=lerobot/pi0 \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.push_to_hub=false \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.gradient_checkpointing=true \
  --policy.use_moe=true \
  --policy.use_diversity_loss=true \
  --policy.moe_num_experts=8 \
  --policy.moe_top_k=2 \
  --policy.moe_expert_intermediate_size=null \
  --policy.moe_lambda_orth=0.05 \
  --policy.moe_lambda_disc=0.02 \
  --policy.moe_load_balance_weight=0.01 \
  --dataset.repo_id=lerobot/libero_10 \
  --batch_size=64 \
  --num_workers=4 \
  --steps=20000 \
  --save_freq=2000 \
  --seed=1006 \
  --output_dir="$OUTDIR" \
  --job_name=pi0_moe_v6_diversity_h100 \
  --wandb.enable=true \
  --wandb.mode=offline \
  --wandb.project=vla-moe-diversity \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.wrist_image": "observation.images.camera2"}'
