#!/bin/bash
# Run this on the Adroit LOGIN NODE (has internet access)
# Compute nodes have NO internet — everything must be pre-cached here.

set -euo pipefail

export HF_HOME=/scratch/network/rj2807/cache/huggingface
export HF_LEROBOT_HOME=/scratch/network/rj2807/lerobot_data
mkdir -p "$HF_HOME" "$HF_LEROBOT_HOME"

# Activate the project venv
source /scratch/network/rj2807/vla-moe-diversity/.venv/bin/activate

echo "=== Downloading SmolVLA base checkpoint ==="
python -c "from huggingface_hub import snapshot_download; snapshot_download('lerobot/smolvla_base')"

echo "=== Downloading PI0 base checkpoint ==="
python -c "from huggingface_hub import snapshot_download; snapshot_download('lerobot/pi0')"

# SmolVLA loads its VLM backbone separately at construction time via
# AutoModelForImageTextToText.from_pretrained('HuggingFaceTB/SmolVLM2-500M-Video-Instruct')
# (see src/lerobot/policies/smolvla/smolvlm_with_expert.py). Pre-cache it too —
# without this the policy build fails on offline compute nodes with LocalEntryNotFoundError.
echo "=== Downloading SmolVLM2 backbone (VLM weights + processor) ==="
python -c "from huggingface_hub import snapshot_download; snapshot_download('HuggingFaceTB/SmolVLM2-500M-Video-Instruct')"

echo "=== Downloading LIBERO-10 dataset ==="
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; LeRobotDataset('lerobot/libero_10')"

# LeRobotDataset downloads into $HF_LEROBOT_HOME/hub/datasets--<org>--<name>/snapshots/<hash>/
# but on subsequent runs the loader looks at $HF_LEROBOT_HOME/<repo_id> first and only
# falls back via a network call to resolve the revision — which fails under HF_HUB_OFFLINE.
# Symlink the snapshot to the expected location so offline runs hit it directly.
echo "=== Linking snapshot into $HF_LEROBOT_HOME/lerobot/libero_10 ==="
SNAPSHOT_DIR=$(ls -d "$HF_LEROBOT_HOME"/hub/datasets--lerobot--libero_10/snapshots/*/ | head -n1)
SNAPSHOT_DIR="${SNAPSHOT_DIR%/}"
mkdir -p "$HF_LEROBOT_HOME/lerobot"
ln -sfn "$SNAPSHOT_DIR" "$HF_LEROBOT_HOME/lerobot/libero_10"

echo "=== Cache setup complete ==="
echo "HF_HOME=$HF_HOME"
echo "HF_LEROBOT_HOME=$HF_LEROBOT_HOME"
ls -lh "$HF_HOME"
ls -lh "$HF_LEROBOT_HOME"
