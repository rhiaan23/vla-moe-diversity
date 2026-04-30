#!/bin/bash
# v7: structural specialization push.
# v6 confirmed: with disc loss bug fixed, entropy still pinned at log2(8) ≈ 3.0
# and cross-task sim ~0.93 across 5 checkpoints. Diversity loss alone does not
# break sparse-upcycling clone symmetry under top-2 routing + load balance.
#
# v7 attacks the structural reasons experts stay clones:
#   - top_k=1            (was 2): kill averaging that blurs experts
#   - load_balance=1e-4  (was 1e-2): 100x weaker uniformity push
#   - lambda_disc=0.5    (was 0.02): 25x stronger disc gradient (now that the
#                        cooperative-CE fix actually trains the discriminator)
#   - lambda_orth=0.0    (was 0.05): proven zero contribution in v6_no_orth control
#   - moe_init_noise=0.1 (was 0.01): break clone symmetry harder at init
#
# Runs on della-stellato GPU 1 (H100 NVL 95GB). v6_diversity continues on GPU 0
# as the "weak diversity loss" baseline for the paper.
set -uo pipefail

REPO=/scratch/gpfs/FHEIDE/rj2807/vla-moe-diversity
LOGDIR=/scratch/gpfs/FHEIDE/rj2807/logs
OUTDIR=/scratch/gpfs/FHEIDE/rj2807/outputs/pi0_moe_v7_specialize_h100
mkdir -p "$LOGDIR"

export HF_LEROBOT_HOME=/scratch/gpfs/FHEIDE/rj2807/lerobot_data
export HF_HOME=/scratch/gpfs/FHEIDE/rj2807/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DIR=/scratch/gpfs/FHEIDE/rj2807/cache/wandb-home
export CUDA_VISIBLE_DEVICES=1

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
  --policy.moe_top_k=1 \
  --policy.moe_expert_intermediate_size=null \
  --policy.moe_init_noise=0.1 \
  --policy.moe_lambda_orth=0.0 \
  --policy.moe_lambda_disc=0.5 \
  --policy.moe_load_balance_weight=0.0001 \
  --dataset.repo_id=lerobot/libero_10 \
  --batch_size=64 \
  --num_workers=4 \
  --steps=20000 \
  --save_freq=1000 \
  --seed=1008 \
  --output_dir="$OUTDIR" \
  --job_name=pi0_moe_v7_specialize_h100 \
  --wandb.enable=true \
  --wandb.mode=offline \
  --wandb.project=vla-moe-diversity \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.wrist_image": "observation.images.camera2"}'
