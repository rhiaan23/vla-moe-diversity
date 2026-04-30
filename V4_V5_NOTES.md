# v4 / v5 Summary and Direction Check

## v4 — Layer-conditioned diversity (incremental fix)

**What changed vs v3.** Same Pi0 + sparse-upcycled MoE-FFN architecture (8
experts per layer × 18 layers, top-2 token-level routing). Two surgical
fixes to the diversity head only:

- `ExpertDiscriminator` (`src/lerobot/policies/smolvla/moe.py:235`) now
  takes `num_layers` and concatenates a per-token layer embedding (16-dim)
  to its input. Previously, "expert 5 in layer 3" and "expert 5 in
  layer 7" shared the same target class even though they're independent
  submodules.
- `compute_orthogonality_loss` is now computed *within each layer's
  expert set* and averaged across layers, rather than pooling per-`expert_id`
  means across layers (which compared vectors from different residual
  streams).

**Training run.** `slurm/run_training_v4_h100.sh`, seed 1002, output
`outputs/pi0_moe_upcycled_v4_h100/`. 20K steps in 25h33m on H100 NVL.

**Numerical result.**

| step  | total loss | flow_matching | moe_orth | moe_disc | moe_lb |
|-------|-----------:|--------------:|---------:|---------:|-------:|
| 200   | 1.51       | 1.47          | 0.0025   | 0        | 0.036  |
| 11400 | 0.28       | 0.26          | 0.0006   | 0        | 0.020  |
| 20000 | **0.283**  | —             | —        | 0        | —      |

v3 final was 0.282. **v4 didn't move the needle on training loss.** The
disc-loss-is-0 observation in v3 turned out to be a logging artifact
(the two CE terms cancel as scalars even though gradients flow), so
there was nothing to "fix" by layer-conditioning the discriminator
beyond making the labels less degenerate.

## v5 — Whole-expert architecture (new direction)

**What changed.** This is a structural rewrite, not a patch.

- `WholeExpertMoELayer` (`src/lerobot/policies/smolvla/moe.py`):
  shared frozen pretrained FFN + N LoRA-rank-16 deltas per FFN layer.
  Routing is *not* per-token; the per-sample assignment is supplied
  externally by the parent model.
- `WholeExpertRouter`: 3-layer MLP (width 256, GELU). Input is the
  mean-pooled embedded prefix (image patches + language token
  embeddings, both available right after `embed_prefix` so we get
  vision + language context without a second VLM pass) concatenated
  with the projected state vector. Output: top-2 over 16 experts,
  Mixtral-style renormalised weights.
- `PI0Pytorch._route_whole_experts` runs the router once per
  forward / once per denoising loop and broadcasts the (B, top_k)
  assignment to every action-expert layer. **The same expert index is
  used for a sample across all 18 layers**, which is what makes
  "expert k" a coherent skill.
- Diversity loss replaced with `compute_lora_orthogonality_loss`:
  applies every expert's ΔW = (α/r)·B·A to a fresh batch of 64 random
  probe vectors, penalises squared cosine similarity between expert
  response vectors. Averaged across (layer × {gate, up, down}). Cheap;
  doesn't require running unused experts on real data. The
  per-expert-output discriminator is gone in this mode.

**Run.** `slurm/run_training_v5_h100.sh`, seed 1003, 16 experts top-2,
20K steps. Currently in flight on the H100; routes per sample, so
expert utilisation comes from the global router's load-balance loss
rather than per-layer token routing.

**Why this is structurally different.** v3/v4 had 18 × 8 = 144
independent FFN sub-modules with no coherent "expert k". v5 has 16
LoRA adapters per FFN, but the per-sample routing decision is *shared
across all layers*, so "expert k" actually corresponds to one identity
that lives across the depth of the action expert. This is the
mixture-of-policies / Soft Modular Networks formulation, not Mixtral.

## Is this the right direction?

**Honest answer: it's the right *architectural* direction, but it's
attacking the wrong question relative to the original ICML 2026
Compositional Learning workshop framing.**

A few things to be candid about:

1. **The deadline (2026-04-24 AOE) has already passed.** Today is
   2026-04-27. v3/v4/v5 are now post-hoc work, not submission work.
   Whatever we ship is for follow-on, not the workshop unless the
   organisers reopen review.

2. **Training loss has been the wrong yardstick the entire time.** All
   variants (baseline, MoE, MoE+diversity, v3, v4) converge to roughly
   the same ~0.28 flow-matching loss on LIBERO-10 because the dataset
   isn't large or hard enough to separate them. The original
   hypothesis — *"diversity-aware MoE generalises better and adapts
   faster"* — lives in the held-out evals (`eval_gen_*.slurm` on
   LIBERO Goal/Object/Spatial) and the few-shot adaptation runs
   (`finetune_gen_*.slurm`). Those have not been run on the v3/v4
   checkpoints. Until they are, no v3/v4/v5 architectural change can
   be claimed to "help" or "not help" on the actual task.

3. **The v3 → v4 patch was scientifically near-zero-effect.** The
   discriminator-treats-layers-the-same critique was real, but the
   numerical impact on training loss was nil, and the disc-loss-is-0
   observation that triggered the investigation was a forward-pass
   cancellation, not a bug in the formulation.

4. **v5 is a different paper, not an improvement on the original.**
   It's not "MoE-FFN action expert with a fixed diversity loss" — it's
   "mixture of N action-expert *policies* selected by a per-sample
   router." That's a clean and defensible architecture (and arguably
   a better fit for the "skills" framing than the original was), but
   calling it a v-bump implies continuity that isn't really there.
   The compelling pitch for v5 is *expert specialisation by task
   semantics*, which can only be shown by:
   - per-task expert utilisation (which task suite triggers which
     expert) on held-out evals
   - few-shot adaptation gains where adapting only the dominant
     expert beats adapting the whole model

5. **What would validate the original hypothesis** (and what nobody
   has done yet):
   - Run `eval_task_perf_*.slurm` on the v3/v4 checkpoints to get
     LIBERO-10 success rate (not loss).
   - Run `eval_gen_*.slurm` on the three held-out suites.
   - Run `finetune_gen_*.slurm` few-shot ablations.
   - Plot expert utilisation per task type — actually inspect whether
     experts specialise.

   None of those depend on the v4/v5 changes; the dense baseline + v3
   checkpoints are enough to either kill or vindicate the original
   diversity-loss hypothesis.

**Recommendation.** Once v5 finishes (~25 h), don't immediately spin
v6. Instead:

1. Run the held-out + finetune evals on baseline, v3, v4, v5
   checkpoints in parallel. That's the actual answer to "does
   diversity-aware MoE help."
2. For v5 specifically: log per-task expert assignments (router top-1
   per sample, grouped by task suite). If experts cleanly specialise,
   that's the v5 paper. If they don't, the whole-expert architecture
   is no better than uniform routing and we learned something useful.
3. If evals show v3/v4 ≈ baseline on success rate, the conclusion is
   "diversity-loss-as-formulated doesn't help on LIBERO" — that's a
   negative result worth writing up with Kim 2026's critique as
   context.

The core risk is that we keep iterating on the architecture and never
answer the actual research question. The architecture is now interesting
enough; the next round of compute should go to evaluation, not training.
