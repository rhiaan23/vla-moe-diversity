#!/bin/bash
# v3: Pi0 MoE with sparse-upcycling init (moe_expert_intermediate_size=null).
# Single-GPU on GPU 0 (GPU 1 free for parallel v2 eval resume or future experiment).
# Rationale: MoE load-balance loss is computed per-rank; without all-reduce of routing
# stats under DDP, expert utilization drifts silently — 39h single-GPU fits the 55h
# deadline, so no reason to take correctness risk. Global batch 64 matches v2.
set -uo pipefail

REPO=/scratch/gpfs/FHEIDE/rj2807/vla-moe-diversity
LOGDIR=/scratch/gpfs/FHEIDE/rj2807/logs
OUTDIR=/scratch/gpfs/FHEIDE/rj2807/outputs/pi0_moe_upcycled_v3
mkdir -p "$LOGDIR"

export HF_LEROBOT_HOME=/scratch/gpfs/FHEIDE/rj2807/lerobot_data
export HF_HOME=/scratch/gpfs/FHEIDE/rj2807/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DIR=/scratch/gpfs/FHEIDE/rj2807/cache/wandb-home
export CUDA_VISIBLE_DEVICES=0

source "$REPO/.venv/bin/activate"

rm -rf "$OUTDIR"

python -m lerobot.scripts.lerobot_train \
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
  --seed=1000 \
  --output_dir="$OUTDIR" \
  --job_name=pi0_moe_upcycled_v3 \
  --wandb.enable=true \
  --wandb.mode=offline \
  --wandb.project=vla-moe-diversity \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.wrist_image": "observation.images.camera2"}'
