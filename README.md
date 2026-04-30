# VLA MoE Diversity Experiments

Mixture-of-Experts action-expert experiments for **Pi0** and **SmolVLA**, built on
top of [LeRobot](https://github.com/huggingface/lerobot). The core question:
*can MoE experts in a VLA action head learn task-conditional skill
specialization on LIBERO-10?*

Six trained variants are versioned `v3 … v8`; each has a self-contained launch
script in `slurm/` that bakes in the exact config used. Architecture details
live in [`V5_ARCHITECTURE.md`](V5_ARCHITECTURE.md).

## Setup

```bash
git clone -b moe-diversity https://github.com/rhiaan23/vla-moe-diversity.git
cd vla-moe-diversity

# Python 3.12+ recommended
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[smolvla,pi0]"
```

## Download Data & Checkpoint

```python
from huggingface_hub import snapshot_download
snapshot_download("lerobot/pi0", local_dir="checkpoints/pi0")             # Pi0 base
snapshot_download("lerobot/smolvla_base", local_dir="checkpoints/smolvla_base")  # optional

from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("lerobot/libero_10")  # auto-cached
```

## Reproduce v5 exactly

The `v5-frozen` git tag pins the exact code that produced the v5 checkpoint.

```bash
git clone -b v5-frozen https://github.com/rhiaan23/vla-moe-diversity.git
cd vla-moe-diversity                           # set up .venv as above
bash slurm/run_training_v5_h100.sh             # 20K steps, ~16h on one H100 NVL
```

To verify the config matched, compare your saved
`outputs/.../checkpoints/000200/pretrained_model/train_config.json` against
the v5 spec table below — every flag should match.

## Experiment matrix

Each version has a launch script that locks in its hyperparameters. Run with
`bash slurm/run_training_v*.sh` (single H100 NVL).

| Version | One-line description | Launch script |
|---|---|---|
| **v3** | Sparse-upcycled experts: deep-copies of pretrained MLP + small init noise. Per-token MoE. | `run_training_v3_h100.sh` |
| **v4** | LoRA experts: shared frozen base FFN + per-expert low-rank delta. Per-token MoE. | `run_training_v4_h100.sh` |
| **v5** | **Whole-expert MoE**: same expert chosen for a sample across *all* layers. LoRA experts, sample-level routing. Produces task-conditional clustering. | `run_training_v5_h100.sh` |
| **v6** | v3-style MoE + cooperative-CE discriminator + functional orthogonality. Diagnoses why diversity loss alone doesn't drive specialization under per-token routing. | `run_training_v6_h100_diversity.sh` |
| **v7** | Structural specialization push on top of v6 (top-1 routing, weak load-balance, large init noise, cranked discriminator). | `run_training_v7_h100_specialize.sh` |
| **v8** | **Whole-expert + sparse upcycle**: combines v5's sample-level routing with v3's full-FFN expert capacity. | `run_training_v8_h100_taskce.sh` |

Eval: `slurm/run_eval_v5_libero10.sh` runs LIBERO-10 episodes; `src/lerobot/scripts/eval_v5_routing.py` computes per-task routing fingerprints (cross-task cosine similarity, expert utilization, entropy).

### v5 spec (whole-expert + LoRA)

| Flag | Value |
|---|---|
| `policy.path` | `lerobot/pi0` |
| `moe_whole_expert` | `true` |
| `moe_num_experts` | `16` |
| `moe_top_k` | `2` |
| `lora_rank` / `lora_alpha` | `16` / `32.0` |
| `use_diversity_loss` | `true` |
| `moe_lambda_orth` | `0.05` (functional orth on LoRA delta directions) |
| `moe_load_balance_weight` | `0.01` |
| `moe_router_hidden_size` / `moe_router_num_layers` | `256` / `3` |
| `seed` | `1003` |
| `dataset.repo_id` | `lerobot/libero_10` |
| `batch_size` / `steps` | `64` / `20000` |
| `dtype` | `bfloat16` |

## MoE config options

All configurable via `--policy.<field>=<value>` (Pi0 fields live in
`src/lerobot/policies/pi0/configuration_pi0.py`; SmolVLA in
`src/lerobot/policies/smolvla/configuration_smolvla.py`).

### Core
| Field | Default | Description |
|---|---|---|
| `use_moe` | `false` | Replace each action-expert layer's MLP with an MoE layer |
| `moe_num_experts` | `8` | Number of experts per layer |
| `moe_top_k` | `2` | Top-k routing |
| `moe_expert_intermediate_size` | `1024` (Pi0) | Per-expert FFN intermediate dim. `null` = use pretrained MLP size (sparse upcycling). Mutually exclusive with `use_lora_experts=true` |
| `moe_load_balance_weight` | `0.01` | Switch-style load-balancing loss weight |
| `moe_init_noise` | `0.01` | Std of Gaussian noise added at sparse-upcycle init |

### LoRA experts
| Field | Default | Description |
|---|---|---|
| `use_lora_experts` | `false` | Each expert = frozen base FFN + per-expert LoRA delta. Preserves pretrained init exactly at step 0 |
| `lora_rank` | `16` | LoRA rank |
| `lora_alpha` | `32.0` | LoRA scaling |
| `lora_dropout` | `0.0` | LoRA dropout |

### Whole-expert MoE (v5, v8)
| Field | Default | Description |
|---|---|---|
| `moe_whole_expert` | `false` | Sample-level routing — same expert assigned across all 18 action-expert layers for a given sample |
| `moe_whole_expert_use_sparse` | `false` | Sparse-upcycled experts (v8) instead of LoRA experts (v5) |
| `moe_router_input` | `"prefix_state"` | Router context: mean-pooled embedded prefix + projected state |
| `moe_router_hidden_size` | `256` | Router MLP hidden dim |
| `moe_router_num_layers` | `3` | Router MLP depth |
| `moe_lora_orth_probes` | `64` | Random-probe count for functional LoRA orthogonality loss |

### Diversity losses
| Field | Default | Description |
|---|---|---|
| `use_diversity_loss` | `false` | Enable orthogonality + discriminability losses |
| `moe_lambda_orth` | `0.05` | Orthogonality loss weight |
| `moe_lambda_disc` | `0.02` | Discriminability loss weight |
| `moe_disc_hidden_size` | `128` | Per-layer discriminator MLP hidden dim |
| `moe_lambda_router_task_ce` | `0.0` | Optional task-supervised CE on router logits (`task_id mod num_experts`). Off by default — used to diagnose whether unsupervised specialization is missing a signal |

## Logged metrics

All MoE runs:
- `moe_lb_loss` — load-balancing auxiliary loss
- `moe_expert_utilization_std` — std of per-expert token fractions (lower = more balanced)

With `use_diversity_loss=true`:
- `moe_orth_loss` — orthogonality between expert outputs (per-layer mean) or LoRA delta directions (whole-expert mode)
- `moe_disc_loss` — discriminator cross-entropy (cooperative single-CE: gradient flows to both discriminator and experts)

With `moe_lambda_router_task_ce > 0`:
- `moe_router_task_ce` — CE between router logits and `task_id mod num_experts`

## Architecture

- **Per-token MoE** (`MoELayer`): each token routes to top-k experts independently. Standard Mixtral-style.
- **Whole-expert MoE** (`WholeExpertMoELayer` + `WholeExpertRouter`): a single per-sample router runs once on `mean_pool(prefix_embs) ‖ state_proj(state)`, picks top-k experts, and broadcasts the assignment to *every* layer. Empirically produces task-conditional cluster structure on LIBERO-10 without explicit task supervision (see `V5_ARCHITECTURE.md` §6).
- **LoRA experts** (`LoRAExpert`): `B=0`, `A∼Kaiming` init so step-0 output equals pretrained FFN output exactly. Cheap (rank=16 → ~70M deltas across 18 layers × 16 experts).
- **Sparse-upcycled experts**: each expert is a deep copy of the pretrained MLP plus Gaussian init noise. Full FFN capacity per expert.

Implementation: `src/lerobot/policies/smolvla/moe.py` (shared module used by both Pi0 and SmolVLA action experts).

## Offline / SLURM usage

Della and similar HPC clusters often have no compute-node internet access:

```bash
# On the login node, pre-cache datasets and weights once
HF_LEROBOT_HOME=/path/to/lerobot_data HF_HOME=/path/to/hf_cache \
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; LeRobotDataset('lerobot/libero_10')"

# In your slurm/run_training_v*.sh
export HF_LEROBOT_HOME=/path/to/lerobot_data
export HF_HOME=/path/to/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

All `slurm/run_training_v*.sh` scripts already follow this pattern and target
`/scratch/gpfs/FHEIDE/rj2807/...` — change those paths for your cluster.

## Original README

The upstream LeRobot README is preserved in
[`README_original.md`](README_original.md).
