#!/bin/bash
# Strict v5 finetune for the "router + VLM LoRA recomposes frozen experts" test.
# Differences from finetune_pi0_v5_router_loravlm_libero01.sh:
#   * --policy.moe_load_balance_weight=0.0  (kills uniform-routing pressure that
#     was forcing the router off the {e5, e11} cluster the v5 baseline learned
#     for tasks 0/1)
#   * --policy.moe_lambda_orth=0.0          (no-op since LoRA experts are frozen,
#     just cleans up logs)
#   * --peft.full_training_modules='["whole_expert_router"]'
#     (drops the IO heads from modules_to_save -> they stay at v5-trained
#     values; only router + VLM-LoRA train)
set -euo pipefail

REPO=/scratch/gpfs/EYSENBACH/ss5822/vla-moe-diversity
CKPT=${REPO}/checkpoints/pi0_moe_whole_v5_h100/checkpoints/020000/pretrained_model
OUT=${REPO}/outputs/finetune_pi0_v5_router_only_libero01

export HF_LEROBOT_HOME=/scratch/gpfs/EYSENBACH/ss5822/data
export HF_HOME=${HOME}/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${REPO}"
rm -rf "${OUT}"

EPISODES='[0,1,4,5,11,18,19,21,22,33,37,52,58,62,68,85,88,105,107,108,110,113,114,121,125,129,130,138,146,152,157,167,168,170,176,190,197,206,207,210,211,212,216,220,224,231,233,235,236,242,247,248,249,252,257,264,267,271,295,301,302,307,309,315,322,323,325,343,346,355,356,362,367,369]'

exec python -m lerobot.scripts.lerobot_train \
  --policy.path="${CKPT}" \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.push_to_hub=false \
  --policy.lora_vlm=true \
  --policy.moe_load_balance_weight=0.0 \
  --policy.moe_lambda_orth=0.0 \
  --peft.method_type=LORA \
  --peft.r=8 \
  --peft.full_training_modules='["whole_expert_router"]' \
  --dataset.repo_id=libero_10 \
  --dataset.root=/scratch/gpfs/EYSENBACH/ss5822/data/libero_10 \
  --dataset.episodes="${EPISODES}" \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.wrist_image": "observation.images.camera2"}' \
  --output_dir="${OUT}" \
  --job_name=ft_pi0_v5_router_only_libero01 \
  --batch_size=8 \
  --num_workers=4 \
  --steps=5000 \
  --save_freq=1000 \
  --log_freq=50 \
  --seed=1003 \
  --wandb.enable=true \
  --wandb.mode=offline \
  --wandb.project=vla-moe-diversity \
  --wandb.notes="v5 finetune, router + VLM-LoRA only (IO heads frozen), lb=0, orth=0, libero tasks 0/1, original prompts"
