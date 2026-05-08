#!/bin/bash
# Strict router-only finetune of Pi0 v5 on LIBERO tasks 0/1.
# Only the whole_expert_router updates. Everything else (PaliGemma, gemma_expert
# incl. all 288 LoRA experts, IO heads, discriminator) is frozen.
#   * No PEFT (no --peft.method_type) -> no VLM LoRA
#   * --policy.freeze_action_expert=true -> freezes gemma_expert subtree
#   * --policy.freeze_io_heads=true     -> freezes state_proj, action_*_proj, action_time_mlp_*
#   * --policy.train_expert_only=true (loaded from v5 saved cfg) keeps PaliGemma frozen
#   * Aux loss weights zeroed so only flow-matching BC drives training.
set -euo pipefail

REPO=/scratch/gpfs/EYSENBACH/ss5822/vla-moe-diversity
CKPT=${REPO}/checkpoints/pi0_moe_whole_v5_h100/checkpoints/020000/pretrained_model
OUT=${REPO}/outputs/finetune_pi0_v5_router_pure_libero01

export HF_LEROBOT_HOME=/scratch/gpfs/EYSENBACH/ss5822/data
export HF_HOME=${HOME}/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Keep wandb run + artifact staging off the home quota.
export WANDB_DIR=/scratch/gpfs/EYSENBACH/ss5822/cache/wandb
export WANDB_CACHE_DIR=/scratch/gpfs/EYSENBACH/ss5822/cache/wandb/cache
export WANDB_DATA_DIR=/scratch/gpfs/EYSENBACH/ss5822/cache/wandb/data
export WANDB_ARTIFACT_DIR=/scratch/gpfs/EYSENBACH/ss5822/cache/wandb/artifacts
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR"

cd "${REPO}"
rm -rf "${OUT}"

EPISODES='[0,1,4,5,11,18,19,21,22,33,37,52,58,62,68,85,88,105,107,108,110,113,114,121,125,129,130,138,146,152,157,167,168,170,176,190,197,206,207,210,211,212,216,220,224,231,233,235,236,242,247,248,249,252,257,264,267,271,295,301,302,307,309,315,322,323,325,343,346,355,356,362,367,369]'

exec python -m lerobot.scripts.lerobot_train \
  --policy.path="${CKPT}" \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.push_to_hub=false \
  --policy.freeze_action_expert=true \
  --policy.freeze_io_heads=true \
  --policy.moe_load_balance_weight=0.0 \
  --policy.moe_lambda_orth=0.0 \
  --dataset.repo_id=libero_10 \
  --dataset.root=/scratch/gpfs/EYSENBACH/ss5822/data/libero_10 \
  --dataset.episodes="${EPISODES}" \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.wrist_image": "observation.images.camera2"}' \
  --output_dir="${OUT}" \
  --job_name=ft_pi0_v5_router_pure_libero01 \
  --batch_size=8 \
  --num_workers=4 \
  --steps=5000 \
  --save_freq=1000 \
  --log_freq=50 \
  --seed=1003 \
  --wandb.enable=true \
  --wandb.mode=offline \
  --wandb.disable_artifact=true \
  --wandb.project=vla-moe-diversity \
  --wandb.notes="v5 finetune, router-only (no VLM LoRA), libero tasks 0/1, original prompts"
