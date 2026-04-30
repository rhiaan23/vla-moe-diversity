#!/bin/bash
# v5 H100: whole-expert MoE on Pi0.
#  - 16 LoRA-adapted action experts share a frozen base FFN
#  - Per-sample top-2 router (Mixtral-style renormalised), same expert
#    assignment for a sample across ALL action-expert layers
#  - Router input: pooled embedded prefix (image patches + lang tokens)
#    + projected state vector. 3-layer MLP with width 256.
#  - Diversity loss = functional orthogonality of LoRA delta directions
#    (random-probe based), no discriminator.
#  - Baseline init: LoRA B=0 → expert output exactly equals pretrained
#    FFN at step 0.
set -uo pipefail

REPO=/scratch/gpfs/FHEIDE/rj2807/vla-moe-diversity
LOGDIR=/scratch/gpfs/FHEIDE/rj2807/logs
OUTDIR=/scratch/gpfs/FHEIDE/rj2807/outputs/pi0_moe_whole_v5_h100
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
  --policy.moe_whole_expert=true \
  --policy.moe_num_experts=16 \
  --policy.moe_top_k=2 \
  --policy.lora_rank=16 \
  --policy.lora_alpha=32.0 \
  --policy.use_diversity_loss=true \
  --policy.moe_lambda_orth=0.05 \
  --policy.moe_load_balance_weight=0.01 \
  --policy.moe_router_hidden_size=256 \
  --policy.moe_router_num_layers=3 \
  --policy.moe_lora_orth_probes=64 \
  --dataset.repo_id=lerobot/libero_10 \
  --batch_size=64 \
  --num_workers=4 \
  --steps=20000 \
  --save_freq=2000 \
  --seed=1003 \
  --output_dir="$OUTDIR" \
  --job_name=pi0_moe_whole_v5_h100 \
  --wandb.enable=true \
  --wandb.mode=offline \
  --wandb.project=vla-moe-diversity \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.wrist_image": "observation.images.camera2"}'
