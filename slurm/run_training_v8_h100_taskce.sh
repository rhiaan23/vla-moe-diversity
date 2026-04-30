#!/bin/bash
# v8 (REWRITTEN after seeing v5 eval): whole-expert routing + sparse-upcycled
# experts. v5 eval revealed:
#   - Whole-expert sample-level routing produced clean task-conditional
#     specialization on its own (cluster {0,1,4,6,7}↔{2,3,5,8,9}, no
#     supervision needed) — the routing recipe is correct.
#   - But v5 only hit 7.5% success with 6 dead experts: LoRA rank=16 doesn't
#     give experts enough capacity to actually do the per-cluster work.
#   - v3 confirmed: full-FFN sparse-upcycled experts retain pretrained
#     capability (no init-gap), but used per-token MoE (no task routing).
#
# v8 = v5 routing recipe + v3 expert capacity:
#   - moe_whole_expert=true             (sample-level routing across all layers)
#   - moe_whole_expert_use_sparse=true  (NEW: deep-copy MLP per expert, not LoRA)
#   - moe_num_experts=8                 (down from v5's 12 — fewer dead experts)
#   - moe_top_k=2                       (mixing two experts per sample)
#   - moe_init_noise=0.1                (break clone symmetry harder than v3's 0.01)
#   - moe_lambda_orth=0.0               (LoRA-orth doesn't apply to sparse)
#   - moe_lambda_disc=0.0               (no per-layer collected outputs in WE mode)
#   - moe_load_balance_weight=0.01      (standard switch-style)
#   - moe_lambda_router_task_ce=0.0     (NOT used: v5 proved unsupervised
#                                        whole-expert routing already specializes)
#
# Hypothesis: full-capacity experts + sample-level routing → both
# task-conditional specialization AND task success.
#
# Runs on della-stellato GPU 1 (H100 NVL).
set -uo pipefail

REPO=/scratch/gpfs/FHEIDE/rj2807/vla-moe-diversity
LOGDIR=/scratch/gpfs/FHEIDE/rj2807/logs
OUTDIR=/scratch/gpfs/FHEIDE/rj2807/outputs/pi0_moe_v8_we_sparse_h100
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
  --policy.moe_whole_expert=true \
  --policy.moe_whole_expert_use_sparse=true \
  --policy.use_diversity_loss=false \
  --policy.moe_num_experts=8 \
  --policy.moe_top_k=2 \
  --policy.moe_init_noise=0.1 \
  --policy.moe_lambda_orth=0.0 \
  --policy.moe_lambda_disc=0.0 \
  --policy.moe_lambda_router_task_ce=0.0 \
  --policy.moe_load_balance_weight=0.01 \
  --dataset.repo_id=lerobot/libero_10 \
  --batch_size=64 \
  --num_workers=4 \
  --steps=20000 \
  --save_freq=1000 \
  --seed=1009 \
  --output_dir="$OUTDIR" \
  --job_name=pi0_moe_v8_we_sparse_h100 \
  --wandb.enable=true \
  --wandb.mode=offline \
  --wandb.project=vla-moe-diversity \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.wrist_image": "observation.images.camera2"}'
