# VLA MoE Diversity Experiments

Mixture-of-Experts action expert experiments for SmolVLA, built on top of [LeRobot](https://github.com/huggingface/lerobot).

## Setup

```bash
# Clone
git clone https://github.com/tharunkumartk/vla-moe-diversity.git
cd vla-moe-diversity

# Create venv and install (requires Python 3.12+)
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[smolvla]"
```

## Download Data & Checkpoint

```python
# Download SmolVLA base checkpoint
from huggingface_hub import snapshot_download
snapshot_download("lerobot/smolvla_base", local_dir="checkpoints/smolvla_base")

# Download LIBERO-10 dataset (auto-cached by LeRobot)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("lerobot/libero_10")
```

## Experiments

There are four configs: baseline, single-expert MoE, standard MoE, and MoE with diversity loss.

### Baseline (no MoE)

```bash
python -m lerobot.scripts.lerobot_train \
  --policy.path=checkpoints/smolvla_base \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/libero_10 \
  --batch_size=32 \
  --steps=50000 \
  --output_dir=outputs/baseline \
  --wandb.enable=true --wandb.project=vla-moe-diversity \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.wrist_image": "observation.images.camera2"}'
```

### Experiment A: Standard MoE

8 experts, top-2 routing, load-balancing loss. Parameter-matched to baseline (~100M trainable).

```bash
python -m lerobot.scripts.lerobot_train \
  --policy.path=checkpoints/smolvla_base \
  --policy.push_to_hub=false \
  --policy.use_moe=true \
  --policy.use_diversity_loss=false \
  --dataset.repo_id=lerobot/libero_10 \
  --batch_size=32 \
  --steps=50000 \
  --output_dir=outputs/moe_standard \
  --wandb.enable=true --wandb.project=vla-moe-diversity \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.wrist_image": "observation.images.camera2"}'
```

### Experiment A0: Single-Expert MoE Ablation

Uses the MoE codepath with exactly one expert, keeping the same per-expert size as the multi-expert runs. Since there is only one expert, routing uses `top_k=1`.

```bash
python -m lerobot.scripts.lerobot_train \
  --policy.path=checkpoints/smolvla_base \
  --policy.push_to_hub=false \
  --policy.use_moe=true \
  --policy.use_diversity_loss=false \
  --policy.moe_num_experts=1 \
  --policy.moe_top_k=1 \
  --dataset.repo_id=lerobot/libero_10 \
  --batch_size=32 \
  --steps=50000 \
  --output_dir=outputs/moe_single_expert \
  --wandb.enable=true --wandb.project=vla-moe-diversity \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.wrist_image": "observation.images.camera2"}'
```

### Experiment B: MoE + Diversity Objective

Same MoE, plus orthogonality loss (pushes expert outputs to be orthogonal) and discriminability loss (rewards experts for producing distinguishable outputs).

```bash
python -m lerobot.scripts.lerobot_train \
  --policy.path=checkpoints/smolvla_base \
  --policy.push_to_hub=false \
  --policy.use_moe=true \
  --policy.use_diversity_loss=true \
  --dataset.repo_id=lerobot/libero_10 \
  --batch_size=32 \
  --steps=50000 \
  --output_dir=outputs/moe_diversity \
  --wandb.enable=true --wandb.project=vla-moe-diversity \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.wrist_image": "observation.images.camera2"}'
```

## MoE Config Options

All configurable via `--policy.<field>=<value>`:

| Field | Default | Description |
|-------|---------|-------------|
| `use_moe` | `false` | Enable MoE expert replacement |
| `moe_num_experts` | `8` | Number of experts per layer |
| `moe_top_k` | `2` | Top-k routing |
| `moe_expert_intermediate_size` | `256` | Per-expert FFN intermediate dim (256 matches baseline param count) |
| `moe_load_balance_weight` | `0.01` | Load-balancing loss weight |
| `use_diversity_loss` | `false` | Enable orthogonality + discriminability losses |
| `moe_lambda_orth` | `0.05` | Orthogonality loss weight |
| `moe_lambda_disc` | `0.02` | Discriminability loss weight |
| `moe_disc_hidden_size` | `128` | Discriminator MLP hidden dim |

## Offline / SLURM Usage

If your compute nodes have no internet access, pre-cache everything on the login node, then run with:

```bash
HF_LEROBOT_HOME=/path/to/lerobot_data \
HF_HOME=/path/to/hf_cache \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m lerobot.scripts.lerobot_train ...
```

## Logged Metrics

When `use_moe=true`, these are logged to wandb:
- `moe_lb_loss` — load-balancing auxiliary loss
- `moe_expert_utilization_std` — std of per-expert token fractions (lower = more balanced)

When `use_diversity_loss=true`, additionally:
- `moe_orth_loss` — orthogonality loss between expert outputs
- `moe_disc_loss` — discriminability loss

## Architecture

The action expert is a 16-layer transformer (384 hidden dim). In MoE mode, each layer's FFN is replaced with N smaller SwiGLU experts + a learned router. The VLM backbone (350M params) stays frozen.

See `src/lerobot/policies/smolvla/moe.py` for the MoE implementation.

## Original README

The original LeRobot README is preserved in [README_original.md](README_original.md).
