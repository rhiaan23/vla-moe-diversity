#!/usr/bin/env bash
# Interactive manual-routing for v5: pick the expert(s) for each chunk yourself.
#
# Usage:
#   ./scripts/run_interactive_routing.sh                       # T5 default
#   ./scripts/run_interactive_routing.sh 2                     # libero_10/T2
#   SUITE=libero_goal ./scripts/run_interactive_routing.sh 0   # libero_goal/T0
#   IMAGE_DIR=/tmp/myrun ./scripts/run_interactive_routing.sh 5
#
# At each chunk boundary the script will:
#   - dump the current camera frames to $IMAGE_DIR/chunk_<NNN>_*.png
#   - print the language instruction, what the learned router would have
#     picked, and the expert vocabulary that has been active on this
#     task historically
#   - prompt you to type the top-k for the next chunk and press enter
#
# Input grammar (any line; whitespace ignored around tokens):
#   <empty>          → passthrough (use the learned router for this chunk)
#   "11"             → expert 11 with weight 1.0 (other top-k slot zero-weighted/ignored)
#   "11,5"           → expert 11 and expert 5 with uniform weights (0.5 each)
#   "11:0.7,5:0.3"   → explicit weights
#   "q"              → quit immediately
#
# To view the dumped frames in real time, open a second terminal and run:
#   feh -R 1 $IMAGE_DIR        # (auto-refresh every 1s)
# or:
#   eog $IMAGE_DIR/chunk_000_observation_images_camera1.png &
#
# Each chunk covers 50 env steps (~1 second of robot motion). A LIBERO-10
# episode is ~520 steps so expect ~11 chunks per episode.

set -uo pipefail

REPO=/scratch/gpfs/FHEIDE/rj2807/vla-moe-diversity
CKPT=/scratch/gpfs/FHEIDE/rj2807/outputs/pi0_moe_whole_v5_h100/checkpoints/last/pretrained_model
SUITE=${SUITE:-libero_10}
TASK_ID=${1:-5}
IMAGE_DIR=${IMAGE_DIR:-/tmp/manual_routing}
OUT_BASE=${OUT_BASE:-/scratch/gpfs/FHEIDE/rj2807/outputs/evals/manual_routing/interactive}
OUT_DIR=$OUT_BASE/${SUITE}_t${TASK_ID}_$(date +%H%M%S)

cd "$REPO"
source .venv/bin/activate

export HF_LEROBOT_HOME=/scratch/gpfs/FHEIDE/rj2807/lerobot_data
export HF_HOME=/scratch/gpfs/FHEIDE/rj2807/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MANUAL_IMAGE_DIR=$IMAGE_DIR

# Make sure robosuite's logging private file exists (avoids first-run write).
ROBOSUITE_PRIV=$(python -c "import robosuite, os; print(os.path.join(os.path.dirname(robosuite.__file__), 'macros_private.py'))" 2>/dev/null || true)
[ -n "$ROBOSUITE_PRIV" ] && [ ! -f "$ROBOSUITE_PRIV" ] && echo "FILE_LOGGING_LEVEL = None" > "$ROBOSUITE_PRIV"

mkdir -p "$IMAGE_DIR"
rm -f "$IMAGE_DIR"/chunk_*_*.png 2>/dev/null
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "  interactive routing on $SUITE / task $TASK_ID"
echo "  image dump: $IMAGE_DIR"
echo "  out dir:    $OUT_DIR"
echo "  in another terminal: feh -R 1 $IMAGE_DIR   (auto-refresh)"
echo "============================================================"

python -m scripts.eval_v5_manual_routing \
  --manual_mode=interactive \
  --policy.path="$CKPT" \
  --policy.moe_top_k=2 \
  --env.type=libero \
  --env.task="$SUITE" \
  --env.task_ids="[$TASK_ID]" \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --policy.device=cuda \
  --output_dir="$OUT_DIR" \
  --job_name="interactive_${SUITE}_t${TASK_ID}" \
  --seed=1000 \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'

rc=$?
echo ""
echo "============================================================"
echo "  done (rc=$rc)"
echo "  routing log: $OUT_DIR/routing_log.json"
echo "  videos:      $OUT_DIR/videos/"
echo "  manifest:    $OUT_DIR/manual_routing_manifest.json"
echo ""
echo "  To overlay your routing decisions onto the rendered video:"
echo "    python scripts/annotate_manual_routing_videos.py $OUT_DIR"
echo "============================================================"
