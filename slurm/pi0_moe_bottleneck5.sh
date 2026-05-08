#!/bin/bash
# Pi0 v5 (whole-expert MoE + LoRA) + 5-D prefix bottleneck.
#  - 16 LoRA-adapted action experts share a frozen base FFN
#  - Per-sample top-2 router; same expert chosen across ALL action-expert
#    layers for a given sample (sample-level routing → "skill" semantics)
#  - Diversity loss = functional orthogonality of LoRA delta directions
#  - On top of v5: action experts cross-attend only to a 1-token K/V
#    synthesized per layer from a 5-D projection of the pooled multimodal
#    prefix (image+lang). Bottleneck forces experts to act mostly open-loop,
#    parametrized only by a low-dim control vector from the VLM.
set -eo pipefail

REPO=/scratch/gpfs/EYSENBACH/ss5822/vla-moe-diversity
OUT=${REPO}/outputs/pi0_moe_bottleneck5
LOG_DIR=${REPO}/logs
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/pi0_moe_bottleneck5_$(date +%Y%m%d_%H%M%S).log"

export HF_LEROBOT_HOME=/scratch/gpfs/EYSENBACH/ss5822/data
# $HOME is over its 50G quota; keep HF cache + wandb on scratch.
export HF_HOME=/scratch/gpfs/EYSENBACH/ss5822/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Rolling checkpoints: lerobot_train deletes the prior checkpoint dir BEFORE
# saving the new one, so peak disk usage is one checkpoint, not two.
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
  --dataset.repo_id=libero_10 \
  --dataset.root=/scratch/gpfs/EYSENBACH/ss5822/data/libero_10 \
  --batch_size=64 \
  --num_workers=4 \
  --steps=10000 \
  --save_freq=2000 \
  --log_freq=200 \
  --output_dir="${OUT}" \
  --job_name=pi0_moe_bottleneck5 \
  --seed=1003 \
  --wandb.enable=true \
  --wandb.mode=offline \
  --wandb.disable_artifact=true \
  --wandb.project=vla-moe-diversity \
  --wandb.notes="pi0 v5 (whole-expert MoE, 16 LoRA experts, top-2) + 5-D prefix bottleneck — 10k steps, rolling 2k ckpts" \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.wrist_image": "observation.images.camera2"}' \
  2>&1 | tee "${LOG}"
