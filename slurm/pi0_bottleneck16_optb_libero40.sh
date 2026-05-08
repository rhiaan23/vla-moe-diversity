#!/bin/bash
# Pi0 v5 (whole-expert MoE + LoRA experts) + 16-D prefix bottleneck w/ option (b)
# (state included in z, no state token in suffix), trained on the 40-task
# libero bundle (libero_10 + libero_object + libero_spatial + libero_goal,
# ~1693 episodes / ~273K frames). NO LoRA on the VLM (vs the libero_loravlm
# variant). Hypothesis: more diverse pretraining data → reusable expert
# primitives without needing VLM adaptation.
set -eo pipefail

REPO=/scratch/gpfs/EYSENBACH/ss5822/vla-moe-diversity
OUT=${REPO}/outputs/pi0_bottleneck16_optb_libero40
LOG_DIR=${REPO}/logs
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/pi0_bottleneck16_optb_libero40_$(date +%Y%m%d_%H%M%S).log"

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
  --policy.moe_router_hidden_size=512 \
  --policy.moe_router_num_layers=6 \
  --policy.moe_lora_orth_probes=64 \
  --policy.prefix_bottleneck=true \
  --policy.prefix_bottleneck_dim=16 \
  --policy.prefix_bottleneck_num_tokens=1 \
  --policy.prefix_bottleneck_source=image_lang \
  --policy.prefix_bottleneck_include_state=true \
  --policy.prefix_bottleneck_hidden=512 \
  --policy.prefix_bottleneck_num_layers=6 \
  --dataset.repo_id=libero \
  --dataset.root=/scratch/gpfs/EYSENBACH/ss5822/data/libero \
  --batch_size=64 \
  --num_workers=4 \
  --steps=10000 \
  --save_freq=2000 \
  --log_freq=200 \
  --output_dir="${OUT}" \
  --job_name=pi0_bottleneck16_optb_libero40 \
  --seed=1003 \
  --wandb.enable=true \
  --wandb.mode=offline \
  --wandb.disable_artifact=true \
  --wandb.project=vla-moe-diversity \
  --wandb.notes="pi0 v5 + 16-D bottleneck OPTION B (state in z) on libero 40-task bundle, no LoRA-VLM, router 6L×512, bottleneck-down 6L×512, 10k steps" \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  2>&1 | tee "${LOG}"
