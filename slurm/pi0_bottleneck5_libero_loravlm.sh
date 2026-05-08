#!/bin/bash
# Pi0 v5 + 5-D bottleneck (img+lang only, state in suffix) on the larger
# bundled lerobot/libero dataset (40 tasks: libero_object + libero_spatial +
# libero_goal + libero_10, ~1693 episodes, ~273K frames). Adds LoRA on the
# PaliGemma VLM so vision-language gets to adapt during pretraining.
# Hypothesis: more diverse pretraining → more reusable expert primitives.
set -eo pipefail

REPO=/scratch/gpfs/EYSENBACH/ss5822/vla-moe-diversity
OUT=${REPO}/outputs/pi0_bottleneck5_libero_loravlm
LOG_DIR=${REPO}/logs
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/pi0_bottleneck5_libero_loravlm_$(date +%Y%m%d_%H%M%S).log"

export HF_LEROBOT_HOME=/scratch/gpfs/EYSENBACH/ss5822/data
export HF_HOME=/scratch/gpfs/EYSENBACH/ss5822/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KEEP_ONLY_LAST_CHECKPOINT=1
export WANDB_DIR=/scratch/gpfs/EYSENBACH/ss5822/cache/wandb_runs
export WANDB_CACHE_DIR=/scratch/gpfs/EYSENBACH/ss5822/cache/wandb_cache
export WANDB_ARTIFACT_DIR=/scratch/gpfs/EYSENBACH/ss5822/cache/wandb_artifacts

cd "${REPO}"
rm -rf "${OUT}"

source /usr/share/Modules/init/bash
module load anaconda3/2025.6
source activate lerobot3

exec python -m lerobot.scripts.lerobot_train \
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
  --policy.prefix_bottleneck=true \
  --policy.prefix_bottleneck_dim=5 \
  --policy.prefix_bottleneck_num_tokens=1 \
  --policy.prefix_bottleneck_source=image_lang \
  --policy.lora_vlm=true \
  --peft.method_type=LORA \
  --peft.r=8 \
  --dataset.repo_id=libero \
  --dataset.root=/scratch/gpfs/EYSENBACH/ss5822/data/libero \
  --batch_size=8 \
  --num_workers=4 \
  --steps=10000 \
  --save_freq=2000 \
  --log_freq=200 \
  --output_dir="${OUT}" \
  --job_name=pi0_bottleneck5_libero_loravlm \
  --seed=1003 \
  --wandb.enable=true \
  --wandb.mode=offline \
  --wandb.disable_artifact=true \
  --wandb.project=vla-moe-diversity \
  --wandb.notes="pi0 v5 + 5-D bottleneck (img+lang) + LoRA-VLM on lerobot/libero (40-task bundle), 10k steps" \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  2>&1 | tee "${LOG}"
