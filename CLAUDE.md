# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fork of [HuggingFace LeRobot](https://github.com/huggingface/lerobot) for MoE (Mixture-of-Experts) action expert experiments on SmolVLA and Pi0. The main research contribution replaces the action expert's FFN layers with MoE layers plus a diversity objective (orthogonality + discriminability losses). MoE code lives in `src/lerobot/policies/smolvla/moe.py` and is shared by both SmolVLA and Pi0 policies.

## Setup

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[smolvla]"
```

For development: `uv pip install -e ".[dev,test,smolvla]"`

## Common Commands

### Training
```bash
python -m lerobot.scripts.lerobot_train \
  --policy.path=checkpoints/smolvla_base \
  --policy.use_moe=true --policy.use_diversity_loss=true \
  --dataset.repo_id=lerobot/libero_10 \
  --batch_size=32 --steps=50000 \
  --output_dir=outputs/moe_diversity
```

### Evaluation
```bash
python -m lerobot.scripts.lerobot_eval \
  --policy.path=<checkpoint_path> \
  --env.type=<env_type>
```

### Linting & Formatting
```bash
ruff check src/          # lint
ruff format src/         # format
pre-commit run --all-files  # full pre-commit suite (ruff, typos, bandit, mypy, etc.)
```

### Tests
```bash
pytest tests/                       # all tests
pytest tests/policies/ -k smolvla   # specific policy tests
make test-smolvla-ete-train DEVICE=cuda  # end-to-end training smoke test
```

### SLURM (offline cluster)
SLURM job scripts are in `slurm/`. Run with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` and pre-cache datasets/checkpoints on the login node.

**Resubmission pitfall**: The training script raises `FileExistsError` if `--output_dir` already exists and `resume` is false (the default). Each slurm script includes `rm -rf <output_dir>` before the python command to clear stale output. Note: `--resume=true` does NOT work for fresh starts — WandB requires a previous run ID in the output dir, so it errors if the dir is missing or was cleaned. Only use `--resume=true` when genuinely resuming from an existing checkpoint.

**Pi0 CPU RAM (OOM) pitfall**: Pi0 is a 4B param model (~16GB in CPU RAM). Each dataloader worker forks the main process, triggering copy-on-write page copies of the model. With `num_workers=7` this can consume 100GB+ of CPU RAM. Use `--mem=192G` and `--num_workers=4` for Pi0 jobs. Do not increase `num_workers` or `batch_size` without testing first.

**Active Pi0 experiment variants** (the only ones to run): baseline, moe_single_expert, moe_standard, moe_diversity. The `orth_only` and `no_orth` ablations are not needed.

## Architecture

### Source Layout
- `src/lerobot/` — main package (installed as editable via `pip install -e`)
- `src/lerobot/policies/` — policy implementations (ACT, Diffusion, SmolVLA, Pi0, etc.)
- `src/lerobot/scripts/` — CLI entry points (`lerobot_train.py`, `lerobot_eval.py`, etc.)
- `src/lerobot/configs/` — draccus-based config dataclasses
- `src/lerobot/datasets/` — LeRobot dataset loading/processing

### MoE Research Code (the main focus of this branch)
- `src/lerobot/policies/smolvla/moe.py` — shared MoE implementation: `MoELayer` (router + N SwiGLU experts), `ExpertDiscriminator`, diversity loss functions (`compute_orthogonality_loss`, `compute_diversity_losses`)
- `src/lerobot/policies/smolvla/configuration_smolvla.py` — SmolVLA config with MoE fields (`use_moe`, `moe_num_experts`, `moe_top_k`, `use_diversity_loss`, etc.)
- `src/lerobot/policies/smolvla/modeling_smolvla.py` — SmolVLA policy; integrates MoE layers and diversity losses into forward pass
- `src/lerobot/policies/smolvla/smolvlm_with_expert.py` — VLM backbone + action expert architecture

### Pi0 MoE Integration
- `src/lerobot/policies/pi0/configuration_pi0.py` — PI0Config with MoE fields mirroring SmolVLA (`use_moe`, `moe_num_experts`, `moe_expert_intermediate_size=1024`, etc.)
- `src/lerobot/policies/pi0/modeling_pi0.py` — PI0Policy; uses flow matching for action denoising, MoE replaces action expert FFN layers
- Pi0 uses `gemma_2b` VLM + `gemma_300m` action expert by default; MoE intermediate size is 1024 (vs 256 for SmolVLA) to keep parameter count comparable

### Config System
Uses [draccus](https://github.com/dlwh/draccus) for CLI config parsing. Policy configs are dataclasses registered via `@PreTrainedConfig.register_subclass("name")`; all parameters are set via `--policy.<field>=<value>`.

**Config loading pitfall**: `PreTrainedConfig.from_pretrained()` parses the checkpoint's YAML config with draccus. If the YAML contains fields not present in the current dataclass (e.g., upstream added/removed fields), draccus raises `DecodingError`. Fix by adding the missing fields to the config dataclass or removing stale fields from the checkpoint YAML.

### Key Experiment Variants
| Config | `use_moe` | `use_diversity_loss` | `moe_num_experts` |
|--------|-----------|---------------------|-------------------|
| Baseline | false | false | — |
| Single Expert | true | false | 1 |
| Standard MoE | true | false | 8 |
| MoE + Diversity | true | true | 8 |

These variants apply to both SmolVLA (`--policy.path=checkpoints/smolvla_base`) and Pi0 (`--policy.path=checkpoints/pi0_base`) policies.

## Code Quality

- **Ruff** for linting and formatting (line length 110, Python 3.12+)
- **Pre-commit hooks**: ruff, typos, bandit, mypy, gitleaks, pyupgrade, prettier (markdown)
- Ruff isort config: `known-first-party = ["lerobot"]`
