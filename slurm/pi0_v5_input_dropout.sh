#!/bin/bash
# Run A: v5 architecture (whole-expert MoE, NO prefix bottleneck) with
# per-modality input dropout on (state, image, language). Goal: experts
# learn to be robust to missing modalities and behave more like generic
# primitives, *without* introducing a learned bottleneck/projector that
# would have to bootstrap from random init. Trained on libero_10 to be
# directly comparable to v5_orig (30%) and bottleneck5 non-optb (14%).
set -eo pipefail

REPO=/scratch/gpfs/EYSENBACH/ss5822/vla-moe-diversity
OUT=${REPO}/outputs/pi0_v5_input_dropout
LOG_DIR=${REPO}/logs
mkdir -p "${LOG_DIR}"
LOG=${LOG_DIR}/pi0_v5_input_dropout_$(date +%Y%m%d_%H%M%S).log

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
  --policy.policy_input_dropout=0.1 \
  --dataset.repo_id=libero_10 \
  --dataset.root=/scratch/gpfs/EYSENBACH/ss5822/data/libero_10 \
  --batch_size=64 \
  --num_workers=4 \
  --steps=10000 \
  --save_freq=2000 \
  --log_freq=200 \
  --output_dir="${OUT}" \
  --job_name=pi0_v5_input_dropout \
  --seed=1003 \
  --wandb.enable=true \
  --wandb.mode=offline \
  --wandb.disable_artifact=true \
  --wandb.project=vla-moe-diversity \
  --wandb.notes="Run A v2: v5 (no bottleneck) + per-modality input dropout p=0.1 on {state,image,lang}, libero_10, 10k steps. Reduced from 0.3 (NaN); mask math now in fp32." \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.wrist_image": "observation.images.camera2"}' \
  2>&1 | tee "${LOG}"
