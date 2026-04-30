#!/bin/bash
# train_loop.sh — runs lerobot_train for a Pi0 variant, auto-resumes on crash/timeout.
# Usage: train_loop.sh <variant> [gpu_id]
# Variants: baseline | moe_single_expert | moe_standard | moe_diversity
#
# On first run: fresh start (wipes OUTPUT_DIR if it exists).
# On subsequent runs: resumes from OUTPUT_DIR/checkpoints/last via --config_path.
# Loop exits when training hits --steps target (touches .done) or the process
# returns 0 without writing a checkpoint.

set -u
set -o pipefail

VARIANT="${1:?variant required}"
GPU_ID="${2:-0}"
TARGET_STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
LOG_FREQ="${LOG_FREQ:-100}"

REPO=/scratch/gpfs/FHEIDE/rj2807/vla-moe-diversity
OUTPUT_DIR="/scratch/gpfs/FHEIDE/rj2807/outputs/pi0_${VARIANT}_v2"
LAST_CFG="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json"
DONE_MARKER="$OUTPUT_DIR/.done"

source "$REPO/.venv/bin/activate"
export HF_LEROBOT_HOME=/scratch/gpfs/FHEIDE/rj2807/lerobot_data
export HF_HOME=/scratch/gpfs/FHEIDE/rj2807/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export WANDB_DIR=/scratch/gpfs/FHEIDE/rj2807/cache/wandb
mkdir -p /scratch/gpfs/FHEIDE/rj2807/cache/wandb

case "$VARIANT" in
    baseline)
        VARIANT_FLAGS=()
        ;;
    moe_single_expert)
        VARIANT_FLAGS=(
            --policy.use_moe=true
            --policy.use_lora_experts=true
            --policy.moe_num_experts=1
            --policy.moe_top_k=1
        )
        ;;
    moe_standard)
        VARIANT_FLAGS=(
            --policy.use_moe=true
            --policy.use_lora_experts=true
            --policy.moe_num_experts=8
            --policy.moe_top_k=2
        )
        ;;
    moe_diversity)
        VARIANT_FLAGS=(
            --policy.use_moe=true
            --policy.use_lora_experts=true
            --policy.use_diversity_loss=true
            --policy.moe_num_experts=8
            --policy.moe_top_k=2
        )
        ;;
    *)
        echo "Unknown variant: $VARIANT" >&2
        exit 2
        ;;
esac

COMMON_FLAGS=(
    --policy.path=lerobot/pi0
    --policy.device=cuda
    --policy.dtype=bfloat16
    --policy.push_to_hub=false
    --policy.train_expert_only=true
    --policy.freeze_vision_encoder=true
    --policy.gradient_checkpointing=true
    --dataset.repo_id=lerobot/libero_10
    --batch_size="$BATCH_SIZE"
    --num_workers=4
    --steps="$TARGET_STEPS"
    --save_freq="$SAVE_FREQ"
    --log_freq="$LOG_FREQ"
    --output_dir="$OUTPUT_DIR"
    --wandb.enable=true
    --wandb.mode=offline
    --wandb.project=vla-moe-diversity
    --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.wrist_image":"observation.images.camera2"}'
)

ITER=0
while true; do
    ITER=$((ITER + 1))
    echo "[$(date '+%F %T')] [loop iter=$ITER variant=$VARIANT gpu=$GPU_ID] starting"

    if [ -f "$DONE_MARKER" ]; then
        echo "[loop] done marker present — exiting"
        break
    fi

    if [ -f "$LAST_CFG" ]; then
        echo "[loop] resuming from $LAST_CFG"
        python -m lerobot.scripts.lerobot_train \
            --config_path="$LAST_CFG" \
            --resume=true
        EXIT=$?
    else
        if [ -d "$OUTPUT_DIR" ]; then
            echo "[loop] no checkpoint found — wiping stale $OUTPUT_DIR for a clean start"
            rm -rf "$OUTPUT_DIR"
        fi
        python -m lerobot.scripts.lerobot_train \
            "${COMMON_FLAGS[@]}" \
            "${VARIANT_FLAGS[@]}"
        EXIT=$?
    fi

    echo "[$(date '+%F %T')] [loop iter=$ITER] exited with code $EXIT"

    # Did we reach the target?
    LAST_STEP=""
    if [ -d "$OUTPUT_DIR/checkpoints" ]; then
        LAST_STEP=$(ls -d "$OUTPUT_DIR"/checkpoints/[0-9]* 2>/dev/null \
            | sort -V | tail -1 | xargs -n1 basename 2>/dev/null | sed 's/^0*//')
    fi

    if [ -n "$LAST_STEP" ] && [ "$LAST_STEP" -ge "$TARGET_STEPS" ] 2>/dev/null; then
        echo "[loop] target step $TARGET_STEPS reached (last=$LAST_STEP) — done"
        touch "$DONE_MARKER"
        break
    fi

    if [ "$EXIT" -eq 0 ] && [ -z "$LAST_STEP" ]; then
        echo "[loop] clean exit with no checkpoint written — aborting"
        break
    fi

    echo "[loop] not done (last=$LAST_STEP target=$TARGET_STEPS) — retrying in 30s"
    sleep 30
done

echo "[$(date '+%F %T')] [loop] finished variant=$VARIANT"
