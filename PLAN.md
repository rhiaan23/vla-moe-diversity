# Current Plan — Pi0 MoE v3 (Sparse Upcycling)

> Status snapshot: 2026-04-24. ICML 2026 Compositional Learning workshop deadline: **2026-04-24 AOE**.

## What we're doing right now

**Type of run:** *Fine-tuning* from `lerobot/pi0`, adapting only the action expert. VLM backbone frozen (`--policy.freeze_vision_encoder=true`, `--policy.train_expert_only=true`). 805M of Pi0's 4.25B params are trainable.

**Dataset:** `lerobot/libero_10` (10 in-distribution tasks). LIBERO-Goal/Object/Spatial reserved for held-out generalization eval. 20K-step horizon (shortened from 50K after loss curves plateau), batch 64, bf16, gradient checkpointing.

**What changed for v3:** Switched from `moe_expert_intermediate_size=1024` (random-init `SmallSwiGLUExpert`) to `moe_expert_intermediate_size=null`, which hits the **sparse-upcycling branch at `src/lerobot/policies/smolvla/moe.py:137-143`** — each MoE expert is a `copy.deepcopy(original_mlp)` of the pretrained gemma_300m FFN plus 0.01-scaled Gaussian noise. No LoRA needed; this preserves the pretrained policy at init and closes the loss gap that motivated the LoRA detour in the old v2 plan.

**Headline experiments for the workshop paper:** `v3_h100` (full diversity) vs `v3_orth_only` vs `v3_no_diversity` on the upcycled architecture — directly ablates Kim 2026's claim that the orthogonality loss is ineffective.

## Active / completed v3 runs

| Run | Variant | Steps | Final loss | Notes |
|-----|---------|-------|------------|-------|
| pi0_baseline_v2 | Baseline (pretrained FFN) | 20K | **~0.26** | Reference. |
| pi0_moe_upcycled_v3 | Full diversity (orth+disc) | 12K | ~0.29 | A100 run, shorter. |
| pi0_moe_upcycled_v3_h100 | Full diversity (orth+disc) | 20K | **0.281** | H100 run, primary result. |
| pi0_moe_upcycled_v3_no_diversity | orth=0, disc=0 | *queued* | — | `slurm/run_training_v3_no_diversity.sh`. |
| pi0_moe_upcycled_v3_orth_only | orth=0.05, disc=0 | *queued* | — | `slurm/run_training_v3_orth_only.sh`. |

Sparse upcycling closed the training-loss gap from ~0.11 (v2) to ~0.02 (v3) vs baseline. Open question: does this translate to LIBERO success rate? → pending v3 eval suite.

## v3 eval suite (fisec RTX6000)

Eval scripts for `v3_h100/checkpoints/last`:
- `slurm/eval_task_perf_v3_fisec.slurm` (LIBERO-10, in-dist)
- `slurm/eval_gen_v3_fisec_{goal,object,spatial}.slurm` (generalization)

Adjust SBATCH header for fisec cluster before submitting. Results land in `outputs/evals_v3/moe_upcycled_v3/libero_{10,goal,object,spatial}/eval_info.json`, mirroring the `outputs/evals_v2/` layout.

## Deferred / longer-horizon

1. **LoRA experts** — still not wired in `pi0/`. Sparse upcycling obviates the main motivation (preserving pretrained init at lower param cost), but LoRA would matter if we scale to more experts. Defer.
2. **Few-shot adaptation** (`finetune_gen_*`) — measures "diversity-aware MoE adapts faster." Worth running after the 3-way v3 ablation table is populated.
3. **4th ablation** `v3_diversity_no_orth` (orth=0, disc=0.02) — add if a 3rd H100 frees up.
