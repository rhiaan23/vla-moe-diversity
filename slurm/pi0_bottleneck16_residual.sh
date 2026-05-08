#!/bin/bash
# Run B: 16-D prefix bottleneck (state in z, image+lang in z) but with
# residual state pathway — keep the state token in the suffix even though
# include_state=True — and zero-init the synth K/V projections so the
# bottleneck path is silent at step 0. Net effect: at init, the action
# expert sees state correctly via the pretrained suffix self-attn (the
# residual) and zero contribution from the bottleneck path; as training
# proceeds, the up_k/up_v 4L×512 MLPs learn to add nonzero K/V on top.
# Image+lang only flow through z (no bypass), so the bottleneck still
# constrains the VLM information path the experts see.
set -eo pipefail

REPO=/scratch/gpfs/EYSENBACH/ss5822/vla-moe-diversity
OUT=${REPO}/outputs/pi0_bottleneck16_residual
LOG_DIR=${REPO}/logs
mkdir -p "${LOG_DIR}"
LOG=${LOG_DIR}/pi0_bottleneck16_residual_$(date +%Y%m%d_%H%M%S).log

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
  --policy.prefix_bottleneck_dim=16 \
  --policy.prefix_bottleneck_num_tokens=1 \
  --policy.prefix_bottleneck_source=image_lang \
  --policy.prefix_bottleneck_include_state=true \
  --policy.prefix_bottleneck_keep_state_token=true \
  --policy.prefix_bottleneck_zero_init_upkv=true \
  --policy.prefix_bottleneck_upkv_num_layers=4 \
  --policy.prefix_bottleneck_upkv_hidden=512 \
  --dataset.repo_id=libero_10 \
  --dataset.root=/scratch/gpfs/EYSENBACH/ss5822/data/libero_10 \
  --batch_size=64 \
  --num_workers=4 \
  --steps=10000 \
  --save_freq=2000 \
  --log_freq=200 \
  --output_dir="${OUT}" \
  --job_name=pi0_bottleneck16_residual \
  --seed=1003 \
  --wandb.enable=true \
  --wandb.mode=offline \
  --wandb.disable_artifact=true \
  --wandb.project=vla-moe-diversity \
  --wandb.notes="Run B: 16-D bottleneck (state in z) + residual state token in suffix + zero-init up_kv (4L×512), libero_10, 10k steps" \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.wrist_image": "observation.images.camera2"}' \
  2>&1 | tee "${LOG}"
