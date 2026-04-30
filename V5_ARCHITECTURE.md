# v5 Architecture — Whole-Expert MoE on Pi0

This is a complete rundown of the v5 model that was trained at
`outputs/pi0_moe_whole_v5_h100/`. It describes the architecture exactly as
it exists in the checkpoint, including the **diversity head** (which is
*not* a discriminator in v5 — see §6).

The shapes and counts below were verified against the live checkpoint at
`checkpoints/last/pretrained_model/model.safetensors`.

---

## 1. The base policy

| component | what it is | size |
|---|---|---|
| **PaliGemma VLM** (frozen) | gemma_2b language model + SigLIP vision tower + projector | ~3.0 B params |
| **Action expert** (Gemma-style, trained) | gemma_300m: depth=18, hidden=1024, mlp_dim=4096, 8 heads, head_dim=256 | ~310 M base + LoRA |
| **State / action / time projections** | `state_proj` (32→1024), `action_in_proj` (32→1024), `action_out_proj` (1024→32), `action_time_mlp_in/out` (2048↔1024) | ~70 K |
| **Whole-expert router** (NEW in v5) | MLP, see §3 | 0.86 M |
| **LoRA experts** (288 total: 16 per layer × 18 layers, NEW in v5) | rank-16 deltas attached to each layer's FFN | 70.78 M |
| **Total checkpoint** | | **4.10 B** |

`paligemma_variant=gemma_2b`, `action_expert_variant=gemma_300m`, `dtype=bfloat16`.

The pretrained Pi0 checkpoint is loaded as the base FFN of every action-expert
layer; the experts attach as LoRA *deltas* on top so step 0 reproduces the
pretrained policy exactly.

---

## 2. Where the MoE lives

The action expert is a stack of **18 Gemma decoder layers**, each containing
self-attention + FFN. **Only the FFN is replaced with MoE.** Self-attention
is shared.

For each layer `ℓ ∈ {0..17}`:

```
layer.mlp = WholeExpertMoELayer(
    base_mlp=<frozen pretrained FFN>,        # gate_proj, up_proj, down_proj
    num_experts=16,
    lora_rank=16, lora_alpha=32.0,
    expert_type="lora",
)
```

Inside each `WholeExpertMoELayer`:

- `base_mlp.gate_proj.weight`: (4096, 1024) — frozen
- `base_mlp.up_proj.weight`: (4096, 1024) — frozen
- `base_mlp.down_proj.weight`: (1024, 4096) — frozen
- 16 × `LoRAExpert`, each holding 6 small matrices:
  - `gate_A`: (16, 1024), `gate_B`: (4096, 16)
  - `up_A`:   (16, 1024), `up_B`:   (4096, 16)
  - `down_A`: (16, 4096), `down_B`: (1024, 16)
  - **= 245 K trainable params/expert × 16 experts × 18 layers = 70.78 M**
- B matrices are zero-initialised so `ΔW = (α/r) · B · A = 0` at step 0.

The other action-expert weights (attention, layernorms, embed/output projs) are also trainable.

---

## 2.5. What the bottom MoE FFN actually sees

The "very top" MoE layer (action-expert layer 0, the first one with no MoE
above it) does **not** receive VLM features as input tokens. The VLM enters
through attention, not through the FFN input pipe. Concretely:

### 2.5.1. The suffix-token sequence (raw input to action-expert layer 0)

Built by `embed_suffix` (`modeling_pi0.py:1003-1064`). Shape:
`(B, 1 + chunk_size, 1024) = (B, 51, 1024)`.

- **Token 0 — state token**:
  `state_proj(robot_state)` — `Linear(max_state_dim=32 → 1024)`. A single
  vector summarizing the proprioceptive state (joint angles + gripper).
- **Tokens 1..50 — noised action chunk**:
  ```
  a_emb  = action_in_proj(x_t)                 # Linear(32→1024); x_t = noised actions at flow time t
  t_emb  = sinusoidal(t)                       # 1024-dim
  fused  = action_time_mlp_in([a_emb ; t_emb]) # Linear(2048→1024)
  fused  = SiLU(fused)
  fused  = action_time_mlp_out(fused)          # Linear(1024→1024)
  ```
  Each of the 50 action tokens carries one timestep of the noised action
  chunk, fused with the (shared across the chunk) flow-matching timestep
  embedding.

That's the entire input embedding to the action expert. **No image or
language tokens are present in this sequence.**

### 2.5.2. How the VLM features enter — dual-tower joint attention

`PaliGemmaWithExpertModel` is a *dual-tower* transformer (openpi pattern):
each of the 18 layers has separate Q/K/V/FFN parameters for the prefix
(PaliGemma, width 2048) and the suffix (gemma_300m action expert, width
1024). Within a layer, attention is computed **jointly** over the
concatenated sequence `[prefix_tokens ; suffix_tokens]`:

- Prefix Q/K/V come from PaliGemma weights, applied to image patch
  embeddings (one set per camera, from the SigLIP tower → projector) and
  language token embeddings (LIBERO task prompt).
- Suffix Q/K/V come from gemma_expert weights, applied to the 51
  state+action tokens above.
- The 51 suffix queries attend over the **full concatenated KV cache**
  (prefix + suffix), so they pull in image+language information during
  attention.

Therefore at layer 0:

```
suffix_in_layer_0       = embed_suffix(state, x_t, t)            # (B, 51, 1024) — no VLM yet
suffix_after_attn_0     = SelfAttn_0([prefix ; suffix])[suffix]   # has absorbed VLM via attention
suffix_after_ffn_0      = MoE_FFN_0(suffix_after_attn_0)          # ← this is what layer-0 MoE sees
```

So the **input that layer-0 LoRA experts receive** is `suffix_after_attn_0`:
shape `(B, 51, 1024)`, semantic content = "state and noised-action tokens
that have already cross-attended once to the visual+language prefix." It is
*not* the raw embeddings, and it is *not* VLM features — it is the suffix
residual stream after one round of joint attention.

For deeper layers `ℓ > 0` the input is `suffix_after_attn_ℓ`, which has
absorbed the prefix `ℓ + 1` times via repeated joint attention.

### 2.5.3. Router input vs. expert input — they are different

This is worth flagging because the router does see VLM features directly,
but the experts do not:

| pipe | source | shape |
|---|---|---|
| **Router input** (`ctx`, see §3) | mean-pool of pre-transformer prefix embeddings ⊕ projected state | `(B, 3072)` |
| **Expert input** (`x` per layer, see §4) | suffix residual stream after that layer's joint attention | `(B, 51, 1024)` |

The router pools the *raw* prefix embeddings (output of `embed_prefix`,
before any transformer block). The experts see the *suffix* residual stream
inside each transformer block. Different tensors, different roles: the
router decides "which skill to use for this (image, language, state)
context"; the expert produces a small LoRA correction to the FFN of the
state+action residual stream that has just attended to that context.

---

## 3. The router

**`WholeExpertRouter`** at `policies/smolvla/moe.py:392`. **One** router for the whole policy — not one per layer.

```
input_dim   = 3072  (= 2048 paligemma width + 1024 action-expert width)
hidden_size = 256
num_layers  = 3                ← MLP depth
num_experts = 16
top_k       = 2
```

Architecture (from the checkpoint):

```
net.0: Linear(3072, 256)       ← 786,688 params
net.1: GELU
net.2: Linear(256, 256)        ←  65,792 params
net.3: GELU
net.4: Linear(256, 16)         ←   4,096 params, weight zero-init
```

**Router input** (`PI0Pytorch._route_whole_experts`, line 848):

1. `embed_prefix` produces the prefix tokens: SigLIP image patch embeddings (one set per camera, paligemma width = 2048) + tokenized language embeddings (also 2048-dim). For LIBERO that's two cameras + the task prompt.
2. **Mean-pool** those prefix tokens over the valid (non-pad) positions → `pooled_prefix : (B, 2048)`.
3. **Project the proprioceptive state** (32-dim joint+gripper vector) through `state_proj` → `state_emb : (B, 1024)`.
4. Concatenate: `ctx = [pooled_prefix ‖ state_emb] : (B, 3072)`.
5. Forward `ctx` through the 3-layer MLP → `logits : (B, 16)`.

**Router output**:

- `probs = softmax(logits, dim=-1)`
- `topk_w, topk_idx = topk(probs, k=2)`
- `topk_w` is renormalised so the two chosen weights sum to 1 (Mixtral-style).
- The router **also returns an aux dict** containing `load_balance_loss` and `tokens_per_expert` (used for the LB term).

**Key property of v5 routing**: `_route_whole_experts` is called *once per policy forward / once per denoising step*. The resulting `(top_2_indices, top_2_weights)` is broadcast to **every one of the 18** `WholeExpertMoELayer`s via `set_whole_expert_routing`. So the same expert pair is used across the entire depth of the action expert for a given sample. That's what makes "expert k" a coherent skill rather than 144 unrelated FFN sub-modules.

During inference the router fires once per `sample_actions` call (chunk size 50 → one routing decision per 50 env steps).

---

## 4. The experts (LoRAExpert) — full forward

For one sample routed to top-k experts `{e_a, e_b}` with weights `{w_a, w_b}`:

Per layer ℓ, the FFN output is:

```
base_gate    = base_mlp.gate_proj(x)              # (B, L, 4096), frozen
base_up      = base_mlp.up_proj(x)                # (B, L, 4096), frozen
base_hidden  = SiLU(base_gate) * base_up          # (B, L, 4096)
base_out     = base_mlp.down_proj(base_hidden)    # (B, L, 1024)

# Per expert e in the active set, compute its corrected output:
gate_full  = base_gate + e.gate_delta(x)          # = base_gate + (α/r)·gate_B·gate_A·x
up_full    = base_up   + e.up_delta(x)            #   same shape
hidden_e   = SiLU(gate_full) * up_full
expert_out = base_mlp.down_proj(hidden_e) + e.down_delta(hidden_e)

delta_e    = (expert_out - base_out) * w_e        # weighted correction

ffn_out    = base_out + Σ_e delta_e
```

So the FFN is `base_FFN(x) + Σ_e w_e · (LoRA_FFN_e(x) − base_FFN(x))`. With 2 active experts and renormalised weights summing to 1, the output is a convex combination of the two LoRA-corrected paths around the base. With B-matrix zero init, all LoRA-corrected paths equal the base at step 0, so `ffn_out = base_out` exactly.

**Inputs** to each LoRA expert: the layer's hidden state `x : (B, L, 1024)`.
**Outputs** of each LoRA expert: contributions to `gate`, `up` (each `(M, 4096)` for the M routed samples), and `down` ((M, 1024)). The expert never produces a "logit" of its own — it only adds to the FFN's gate/up/down projections.

---

## 5. Loss aggregation (training)

From `PI0Pytorch._aggregate_moe_losses` (line 878), the training loss in v5 is:

```
loss = flow_matching_loss
     + moe_load_balance_weight * lb_loss        # weight = 0.01
     + moe_lambda_orth         * lora_orth_loss # weight = 0.05
```

where:

- **`flow_matching_loss`** — the standard Pi0 BC objective: predict the velocity field for the noisy action chunk.
- **`lb_loss`** — Switch Transformer load-balance: `E · Σ (frac_assigned_e · mean_prob_e)`. Discourages router collapse onto a few experts.
- **`lora_orth_loss`** — see §6.

There is **no** discriminator term in v5.

---

## 6. The "diversity head" in v5 — and the discriminator question

This is the part that needs to be stated plainly:

> **v5 has NO discriminator.** The `discriminator` field on `PI0Pytorch` is `None` for whole-expert mode (`modeling_pi0.py:772-797`). The checkpoint contains zero `discriminator.*` keys.

The discriminator was the v3/v4 mechanism: a small MLP that took an FFN expert's output and tried to classify which expert produced it; its CE loss was added to encourage experts to be distinguishable. In v5 the architecture changed so that:

1. Routing is per-sample (not per-token), so there's no longer a notion of "this token's output came from expert k".
2. With LoRA experts on a shared base, expert outputs differ only by `(α/r)·B·A·x`, so a discriminator on raw FFN outputs would mostly classify based on the shared base, not the deltas.

So v5 replaces the discriminator with a **parameter-space functional orthogonality loss** on the LoRA deltas themselves. From `compute_lora_orthogonality_loss` (`moe.py:451`):

```
For each layer ℓ in {0..17} × each projection p in {gate, up, down}:
    Sample 64 random probe vectors v_i.
    For each expert e: compute (ΔW_p^e · v_i) — the expert's response to the probe.
    Build M : (16 experts, 64·out_dim) by stacking flattened responses.
    L2-normalize each row.
    Compute pairwise cosine sim S = MM^T : (16, 16).
    Off-diagonal squared-cos-sim is the per-(layer, projection) loss.
Average across all (layer × projection) pairs → scalar.
```

**Inputs**: the LoRA `A` and `B` weights themselves (no real data).
**Outputs**: a scalar in [0, 1] — 0 = experts move random vectors in mutually orthogonal directions, 1 = identical.

Why probes instead of real activations: doesn't require running every expert on every batch (only the top-2 are dispatched in the forward), and measures *functional* not just *parameter* diversity.

---

## 7. So — what about "discriminator accuracy"?

Because there is no discriminator in v5, there is no classification accuracy
to report. The training-time analogue, `train/moe_orth_loss`, plateaued at
~2.5e-5 by step 5K (already very small at init because B=0 means ΔW≈0).
That number is *the* diversity health metric for v5; it stayed small
throughout training, which is consistent with the LoRA deltas remaining
near-orthogonal in their probe responses. It does **not** mean the experts
are interchangeable — see the eval routing data, where the router learned
clear task-conditional clusters (`{e5, e11}` for tasks 0/1/4/6/7;
`{e8, e9, e1}` for tasks 2/3/5/8/9; 6 of 16 experts dead).

If you want a *router* accuracy (which is the closest v5 analogue), that
would be: "given the prefix+state, does the router pick the cluster of
experts that the policy actually trained against for this task?" Since
training used the same router with no task-id supervision, there is no
ground-truth label and "accuracy" isn't well-defined. The clean
specialisation pattern in the eval data is the strongest evidence that
the router learned something useful.

For comparison, **v3 and v4 do have a discriminator** (`ExpertDiscriminator`
in `moe.py:497`). It's a 2-layer MLP that takes an FFN expert's output
vector + a layer embedding (in v4) and outputs `num_experts` logits trained
with cross-entropy against the routed expert id. Those checkpoints would
have a non-trivial accuracy number; this v5 checkpoint does not.

---

## 8. End-to-end forward pass (training)

Here's the full flow for one batch step:

```
1. Image preprocess + tokenize prompt.
2. embed_prefix(images, lang_tokens) → prefix_embs : (B, T_pre, 2048)
3. state, noisy_action_chunk, t = sample()
4. embed_suffix(state, noisy_action_chunk, t) → suffix_embs : (B, T_suf, 1024)
5. _route_whole_experts(prefix_embs, prefix_pad_masks, state):
    pooled = mean-pool(prefix_embs over pad mask)
    state_emb = state_proj(state)
    ctx = [pooled || state_emb]              # (B, 3072)
    top2_idx, top2_w, lb_aux = router(ctx)
    set_whole_expert_routing(top2_idx, top2_w)  # broadcast to all 18 layers
6. PaliGemma forward over prefix produces KV cache for the prefix.
7. Action expert (gemma_300m) forward over suffix using prefix KV; each layer's
   FFN dispatches to the top-2 LoRA experts using (top2_idx, top2_w) above.
8. Predict velocity field ε̂.
9. Loss = ‖ε̂ - target‖² + 0.01 · lb_loss + 0.05 · lora_orth_loss(probes=64).
```

Inference is the same with the diffusion loop replaced by 10 denoising steps;
`_route_whole_experts` runs once per denoising step. Action chunk size 50.

---

## 9. Numerical defaults (v5)

```
moe_num_experts            = 16
moe_top_k                  = 2
lora_rank                  = 16
lora_alpha                 = 32.0       (so scaling = α/r = 2.0)
lora_dropout               = 0.0
moe_router_hidden_size     = 256
moe_router_num_layers      = 3
moe_lora_orth_probes       = 64
moe_load_balance_weight    = 0.01
moe_lambda_orth            = 0.05
moe_init_noise             = 0.01       (used in sparse mode only; v5 is "lora")
moe_whole_expert_use_sparse = false
chunk_size                 = 50
num_inference_steps        = 10
batch_size (training)      = 64
steps                      = 20000
seed                       = 1003
```

---

## 10. Files / call sites

- `src/lerobot/policies/smolvla/moe.py:264` — `WholeExpertMoELayer`
- `src/lerobot/policies/smolvla/moe.py:392` — `WholeExpertRouter`
- `src/lerobot/policies/smolvla/moe.py:451` — `compute_lora_orthogonality_loss`
- `src/lerobot/policies/smolvla/moe.py:497` — `ExpertDiscriminator` (v3/v4 only, NOT used in v5)
- `src/lerobot/policies/pi0/modeling_pi0.py:506-521` — wraps every layer.mlp into `WholeExpertMoELayer`
- `src/lerobot/policies/pi0/modeling_pi0.py:1003-1064` — `embed_suffix`: builds the state+noised-action+timestep token sequence (input to action-expert layer 0)
- `src/lerobot/policies/pi0/modeling_pi0.py:782-788` — instantiates the global router
- `src/lerobot/policies/pi0/modeling_pi0.py:848-876` — `_route_whole_experts` (called once per forward)
- `src/lerobot/policies/pi0/modeling_pi0.py:892-916` — v5 loss aggregation
- `src/lerobot/policies/pi0/configuration_pi0.py:131-141` — config fields added for v5
- `slurm/run_training_v5_h100.sh` — the exact training recipe used
