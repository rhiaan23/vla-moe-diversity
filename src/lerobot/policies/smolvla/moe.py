"""Mixture-of-Experts modules for SmolVLA action expert.

Provides:
- MoELayer: Drop-in replacement for LlamaMLP with N expert copies + top-k router
- LoRAExpert: Per-expert LoRA deltas sharing a frozen pretrained FFN
- ExpertDiscriminator: Small classifier for the diversity objective
- compute_diversity_losses: Orthogonality + discriminability losses
"""

import copy
import math
from contextvars import ContextVar

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Per-batch task index (B,) made available to MoELayer.forward without changing
# any forward signatures. Set by the policy's forward() before running the
# action expert; reset in a finally block. Used to compute a task-supervised
# cross-entropy on the router logits (CE(router_logits, task_id_per_token)),
# which provides a constructive task-conditional signal that pure BC can't.
_current_task_index: ContextVar = ContextVar("_current_task_index", default=None)


def set_current_task_index(task_idx):
    """Set the per-batch task_index tensor (shape (B,)) for the next forward.

    Pass None to clear.
    """
    _current_task_index.set(task_idx)


class SmallSwiGLUExpert(nn.Module):
    """A smaller SwiGLU MLP expert with configurable intermediate size."""

    def __init__(self, hidden_size: int, intermediate_size: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)
        self.act_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LoRAExpert(nn.Module):
    """Per-expert LoRA deltas for a shared frozen SwiGLU FFN.

    Adds rank-`r` low-rank updates to gate_proj, up_proj, and down_proj.
    Standard LoRA init: A ~ kaiming, B = 0 so the delta is 0 at init and
    the MoE output exactly equals the pretrained FFN output on the first
    forward pass.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.gate_A = nn.Linear(hidden_size, rank, bias=False, dtype=dtype)
        self.gate_B = nn.Linear(rank, intermediate_size, bias=False, dtype=dtype)
        self.up_A = nn.Linear(hidden_size, rank, bias=False, dtype=dtype)
        self.up_B = nn.Linear(rank, intermediate_size, bias=False, dtype=dtype)
        self.down_A = nn.Linear(intermediate_size, rank, bias=False, dtype=dtype)
        self.down_B = nn.Linear(rank, hidden_size, bias=False, dtype=dtype)

        for lin in (self.gate_A, self.up_A, self.down_A):
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
        for lin in (self.gate_B, self.up_B, self.down_B):
            nn.init.zeros_(lin.weight)

    def gate_delta(self, x: Tensor) -> Tensor:
        return self.scaling * self.gate_B(self.gate_A(self.dropout(x)))

    def up_delta(self, x: Tensor) -> Tensor:
        return self.scaling * self.up_B(self.up_A(self.dropout(x)))

    def down_delta(self, h: Tensor) -> Tensor:
        return self.scaling * self.down_B(self.down_A(self.dropout(h)))


class MoELayer(nn.Module):
    """Mixture-of-Experts FFN layer.

    Replaces a single LlamaMLP with N expert copies and a learned router.
    Initialized via sparse upcycling: all experts start as copies of the
    original pretrained MLP weights, with small noise to break symmetry.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        original_mlp: nn.Module,
        expert_intermediate_size: int | None = None,
        use_lora_experts: bool = False,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
        init_noise: float = 0.01,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_lora_experts = use_lora_experts
        self.init_noise = init_noise

        # Router: maps hidden states to expert selection logits
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.kaiming_uniform_(self.router.weight, a=1.0)

        self.experts = nn.ModuleList()
        self.base_mlp: nn.Module | None = None

        if use_lora_experts:
            # LoRA mode: shared frozen pretrained FFN + per-expert LoRA delta.
            # base_mlp is a real submodule (stored once), not deep-copied per expert,
            # so checkpoints stay small and all experts use the same pretrained weights.
            self.base_mlp = original_mlp
            for param in self.base_mlp.parameters():
                param.requires_grad = False

            dtype = next(original_mlp.parameters()).dtype
            # Infer intermediate size from the pretrained MLP's gate_proj
            intermediate_size = original_mlp.gate_proj.out_features
            for _ in range(num_experts):
                self.experts.append(
                    LoRAExpert(
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        rank=lora_rank,
                        alpha=lora_alpha,
                        dropout=lora_dropout,
                        dtype=dtype,
                    )
                )
        elif expert_intermediate_size is not None:
            # Smaller experts for parameter-matched comparison (random init)
            dtype = next(original_mlp.parameters()).dtype
            for _ in range(num_experts):
                self.experts.append(SmallSwiGLUExpert(hidden_size, expert_intermediate_size, dtype=dtype))
        else:
            # Sparse upcycling: deep copy the original pretrained MLP
            for _ in range(num_experts):
                expert = copy.deepcopy(original_mlp)
                for param in expert.parameters():
                    param.data += init_noise * torch.randn_like(param.data)
                self.experts.append(expert)

    def forward(self, x: Tensor, collect_expert_outputs: bool = False) -> tuple[Tensor, dict]:
        """
        Args:
            x: (B, L, D) input hidden states
            collect_expert_outputs: if True, return per-expert outputs for diversity losses

        Returns:
            (output, aux_dict) where output is (B, L, D) and aux_dict contains
            load_balance_loss and optionally expert output data.
        """
        B, L, D = x.shape
        input_dtype = x.dtype
        x_flat = x.view(-1, D)  # (N, D) where N = B*L
        N = x_flat.shape[0]

        # Route — cast router to input dtype (experts are already in input dtype)
        router_logits = F.linear(x_flat, self.router.weight.to(input_dtype))  # (N, E)
        router_probs = F.softmax(router_logits, dim=-1)  # (N, E)
        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (N, k)

        # Normalize top-k weights to sum to 1
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Dispatch tokens to experts and combine
        output = torch.zeros_like(x_flat)  # (N, D)
        expert_outputs_for_diversity = [] if collect_expert_outputs else None
        expert_labels_for_diversity = [] if collect_expert_outputs else None

        # In LoRA mode, precompute base gate/up projections once for all tokens.
        # Base is frozen, so this is the only call we need per layer.
        base_gate_all = None
        base_up_all = None
        if self.use_lora_experts:
            base_gate_all = self.base_mlp.gate_proj(x_flat)  # (N, mlp_dim)
            base_up_all = self.base_mlp.up_proj(x_flat)  # (N, mlp_dim)

        for expert_idx, expert in enumerate(self.experts):
            # Find which tokens are routed to this expert (across any of the top-k slots)
            # mask: (N,) bool — True if this expert is in the top-k for that token
            mask = (topk_indices == expert_idx).any(dim=-1)  # (N,)
            if not mask.any():
                continue

            expert_input = x_flat[mask]  # (M, D)

            if self.use_lora_experts:
                gate = base_gate_all[mask] + expert.gate_delta(expert_input)
                up = base_up_all[mask] + expert.up_delta(expert_input)
                hidden = self.base_mlp.act_fn(gate) * up
                expert_out = self.base_mlp.down_proj(hidden) + expert.down_delta(hidden)
            else:
                expert_out = expert(expert_input)  # (M, D)

            # Get the weight for this expert for the selected tokens
            # For each token, find which top-k slot(s) match this expert and sum their weights
            slot_mask = topk_indices[mask] == expert_idx  # (M, k)
            weight = (topk_weights[mask] * slot_mask.float()).sum(dim=-1, keepdim=True)  # (M, 1)

            output[mask] += weight * expert_out

            if collect_expert_outputs:
                expert_outputs_for_diversity.append(expert_out.detach() if False else expert_out)
                expert_labels_for_diversity.append(
                    torch.full((expert_out.shape[0],), expert_idx, device=x.device, dtype=torch.long)
                )

        output = output.view(B, L, D)

        # Load-balancing loss (Switch Transformer style)
        # tokens_per_expert: fraction of tokens dispatched to each expert
        # mean_routing_prob: average router probability for each expert
        tokens_per_expert = torch.zeros(self.num_experts, device=x.device)
        for expert_idx in range(self.num_experts):
            tokens_per_expert[expert_idx] = (topk_indices == expert_idx).any(dim=-1).float().mean()
        mean_routing_prob = router_probs.mean(dim=0)  # (E,)
        load_balance_loss = self.num_experts * (tokens_per_expert * mean_routing_prob).sum()

        aux = {
            "load_balance_loss": load_balance_loss,
            "router_logits": router_logits,
            "tokens_per_expert": tokens_per_expert,
        }

        # Task-supervised router CE: maps task_id mod num_experts to the desired
        # expert and cross-entropies the router logits against it. Provides a
        # constructive task-conditional signal — pure BC + diversity loss has
        # no such signal (confirmed empirically: v6/v7 sim ~0.93–0.98, no task
        # differentiation regardless of disc strength or top-k).
        tid = _current_task_index.get()
        if tid is not None:
            tid_per_token = tid.view(B, 1).expand(B, L).reshape(N).to(router_logits.device)
            target = (tid_per_token % self.num_experts).long()
            aux["task_router_ce"] = F.cross_entropy(router_logits.float(), target)

        if collect_expert_outputs and expert_outputs_for_diversity:
            aux["expert_outputs"] = expert_outputs_for_diversity
            aux["expert_labels"] = expert_labels_for_diversity

        return output, aux


class WholeExpertMoELayer(nn.Module):
    """FFN layer where the expert assignment is supplied externally.

    Unlike `MoELayer` (which has its own per-layer router and routes per
    token), this module accepts a per-sample top-k expert assignment from
    a global router living on the parent model. The same expert index is
    therefore used across every layer of the action expert for a given
    sample, which is what makes "expert k" a coherent skill.

    Implementation modes (selected via `expert_type`):
    - "lora": shared frozen base SwiGLU FFN + per-expert LoRA delta. Each
      forward runs the LoRA-adapted FFN on the subset of samples for that
      expert and accumulates the weighted delta onto the dense base output.
    - "sparse": each expert is a full deep-copy of the pretrained MLP +
      Gaussian init noise (sparse upcycling). No shared base; each expert
      starts at full capacity from the pretrained weights. Combines v5's
      sample-level routing (proven to specialize by task without supervision)
      with v3's full-capacity experts (proven to retain task success).
    """

    def __init__(
        self,
        base_mlp: nn.Module,
        num_experts: int,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
        expert_type: str = "lora",
        init_noise: float = 0.01,
    ):
        super().__init__()
        if expert_type not in ("lora", "sparse"):
            raise ValueError(f"expert_type must be 'lora' or 'sparse', got {expert_type!r}")
        self.num_experts = num_experts
        self.expert_type = expert_type
        dtype = next(base_mlp.parameters()).dtype
        hidden_size = base_mlp.gate_proj.in_features
        intermediate_size = base_mlp.gate_proj.out_features

        if expert_type == "lora":
            self.base_mlp = base_mlp
            for p in self.base_mlp.parameters():
                p.requires_grad = False
            self.experts = nn.ModuleList(
                [
                    LoRAExpert(
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        rank=lora_rank,
                        alpha=lora_alpha,
                        dropout=lora_dropout,
                        dtype=dtype,
                    )
                    for _ in range(num_experts)
                ]
            )
        else:  # sparse
            # No shared base; experts are full deep copies that all start
            # near-identical to the pretrained MLP, with init_noise breaking
            # symmetry. base_mlp is kept as a fallback for the no-routing path
            # and as the source for deep-copies, but its params are frozen
            # (the experts hold the trainable weights).
            self.base_mlp = base_mlp
            for p in self.base_mlp.parameters():
                p.requires_grad = False
            self.experts = nn.ModuleList()
            for _ in range(num_experts):
                expert = copy.deepcopy(base_mlp)
                for p in expert.parameters():
                    p.data += init_noise * torch.randn_like(p.data)
                    p.requires_grad = True
                self.experts.append(expert)
        # Set per forward by parent model; both shape (B, k).
        self._expert_indices: Tensor | None = None
        self._expert_weights: Tensor | None = None

    def set_routing(self, indices: Tensor, weights: Tensor) -> None:
        self._expert_indices = indices
        self._expert_weights = weights

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, D).
        if self._expert_indices is None or self._expert_weights is None:
            return self.base_mlp(x)

        idx = self._expert_indices  # (B, k)
        wts = self._expert_weights  # (B, k)
        # Per-sample weight assigned to each expert (sum of any matching top-k slots).
        # one_hot: (B, k, E); per_expert_weight: (B, E)
        one_hot = F.one_hot(idx, num_classes=self.num_experts).to(wts.dtype)
        per_expert_weight = (wts.unsqueeze(-1) * one_hot).sum(dim=1)

        if self.expert_type == "sparse":
            # Each expert is a full FFN — sum weighted expert outputs directly.
            output = torch.zeros_like(x)
            for e, expert in enumerate(self.experts):
                sample_w = per_expert_weight[:, e]  # (B,)
                mask = sample_w > 0
                if not mask.any():
                    continue
                x_sub = x[mask]
                expert_out = expert(x_sub)
                contribution = expert_out * sample_w[mask].view(-1, 1, 1)
                output = output.index_add(0, mask.nonzero(as_tuple=True)[0], contribution)
            return output

        # LoRA mode: shared frozen base + per-expert delta.
        base_gate = self.base_mlp.gate_proj(x)
        base_up = self.base_mlp.up_proj(x)
        base_hidden = self.base_mlp.act_fn(base_gate) * base_up
        base_out = self.base_mlp.down_proj(base_hidden)

        delta = torch.zeros_like(base_out)
        for e, expert in enumerate(self.experts):
            sample_w = per_expert_weight[:, e]  # (B,)
            mask = sample_w > 0
            if not mask.any():
                continue
            x_sub = x[mask]
            gate_full = base_gate[mask] + expert.gate_delta(x_sub)
            up_full = base_up[mask] + expert.up_delta(x_sub)
            hidden_full = self.base_mlp.act_fn(gate_full) * up_full
            expert_out = self.base_mlp.down_proj(hidden_full) + expert.down_delta(hidden_full)
            d = (expert_out - base_out[mask]) * sample_w[mask].view(-1, 1, 1)
            delta = delta.index_add(0, mask.nonzero(as_tuple=True)[0], d)
        return base_out + delta


class WholeExpertRouter(nn.Module):
    """Global per-sample top-k router for whole-expert MoE.

    Maps a per-sample context vector to a softmax over experts, then takes
    top-k with renormalised weights (Mixtral-style — gradients flow through
    the chosen weights, no Gumbel needed).

    Architecture: `num_layers`-deep MLP with `hidden_size`-width hidden
    layers and GELU activations. The final projection to expert logits is
    zero-initialised so routing starts uniform and the load-balance loss
    spreads it from there.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 2,
        hidden_size: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(max(num_layers - 1, 0)):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.GELU())
            in_dim = hidden_size
        final = nn.Linear(in_dim, num_experts, bias=False)
        nn.init.zeros_(final.weight)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, ctx: Tensor) -> tuple[Tensor, Tensor, dict]:
        # ctx: (B, D) per-sample context vector
        logits = self.net(ctx.float())
        probs = F.softmax(logits, dim=-1)  # (B, E)
        topk_w, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-9)

        # Switch-style load-balance loss
        with torch.no_grad():
            assign = torch.zeros_like(probs)
            assign.scatter_add_(
                1, topk_idx, torch.ones_like(topk_w)
            )
            tokens_per_expert = (assign > 0).float().mean(dim=0)
        mean_routing_prob = probs.mean(dim=0)
        lb_loss = self.num_experts * (tokens_per_expert * mean_routing_prob).sum()
        aux = {
            "load_balance_loss": lb_loss,
            "tokens_per_expert": tokens_per_expert,
            "router_probs": probs,
        }
        return topk_idx, topk_w, aux


def compute_lora_orthogonality_loss(
    layers: list[WholeExpertMoELayer], n_probes: int = 64
) -> Tensor:
    """Functional orthogonality between expert LoRA deltas.

    For each layer and each LoRA-adapted projection (gate, up, down),
    apply every expert's ΔW = (alpha/r) * B @ A to the same set of random
    probe vectors and penalise the squared cosine similarity between
    expert response vectors. Averaged across (layer × projection).

    This measures *functional* diversity (do the experts move random inputs
    in different directions?) without requiring all N experts to actually
    process real data on every batch — much cheaper than running all
    experts during the forward pass.
    """
    if not layers:
        return torch.tensor(0.0)

    device = layers[0].experts[0].gate_A.weight.device
    losses: list[Tensor] = []

    for layer in layers:
        hidden_size = layer.experts[0].gate_A.weight.shape[1]
        intermediate_size = layer.experts[0].down_A.weight.shape[1]
        # Fresh probes per call so the loss isn't degenerate.
        probe_h = torch.randn(n_probes, hidden_size, device=device, dtype=torch.float32)
        probe_m = torch.randn(n_probes, intermediate_size, device=device, dtype=torch.float32)

        for proj, probe in (("gate", probe_h), ("up", probe_h), ("down", probe_m)):
            outs = []
            for ex in layer.experts:
                A = getattr(ex, f"{proj}_A").weight.float()
                B = getattr(ex, f"{proj}_B").weight.float()
                # ΔW @ probe^T = (B @ (A @ probe^T)) → (out, n_probes); flatten.
                out = (probe @ A.T) @ B.T  # (n_probes, out_dim)
                outs.append(out.flatten() * ex.scaling)
            M = torch.stack(outs, dim=0)  # (E, n_probes * out_dim)
            Mn = F.normalize(M, dim=-1, eps=1e-8)
            sim = Mn @ Mn.T
            K = sim.shape[0]
            mask = ~torch.eye(K, dtype=torch.bool, device=sim.device)
            losses.append((sim[mask] ** 2).mean())

    return torch.stack(losses).mean()


class ExpertDiscriminator(nn.Module):
    """Small MLP classifier that predicts which expert produced a given output.

    When `num_layers > 1` a per-layer embedding is concatenated to the input so
    the same `expert_id` in different layers is no longer treated as the same
    class. Without this, expert i in layer A and expert i in layer B share a
    label even though they're independent submodules — the loss optimises an
    arbitrary cross-layer correlation rather than diversity.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_layers: int = 1,
        layer_emb_dim: int = 16,
        disc_hidden_size: int = 128,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_layer_emb = num_layers > 1
        if self.use_layer_emb:
            self.layer_emb = nn.Embedding(num_layers, layer_emb_dim)
            input_size = hidden_size + layer_emb_dim
        else:
            input_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(input_size, disc_hidden_size),
            nn.ReLU(),
            nn.Linear(disc_hidden_size, num_experts),
        )

    def forward(self, x: Tensor, layer_idx: Tensor | int | None = None) -> Tensor:
        """
        Args:
            x: (M, D) expert output vectors
            layer_idx: (M,) long tensor or scalar int. Required when
                `num_layers > 1`; ignored otherwise.
        Returns:
            logits: (M, num_experts)
        """
        if self.use_layer_emb:
            if layer_idx is None:
                raise ValueError("layer_idx must be provided when num_layers > 1")
            if not isinstance(layer_idx, torch.Tensor):
                layer_idx = torch.full((x.shape[0],), layer_idx, dtype=torch.long, device=x.device)
            elif layer_idx.dim() == 0:
                layer_idx = layer_idx.expand(x.shape[0])
            emb = self.layer_emb(layer_idx).to(x.dtype)
            x = torch.cat([x, emb], dim=-1)
        return self.net(x)


def compute_orthogonality_loss(expert_outputs: list[Tensor]) -> Tensor:
    """Penalize cosine similarity between mean expert output vectors.

    Args:
        expert_outputs: list of (M_i, D) tensors, one per active expert

    Returns:
        Scalar loss (higher = more similar experts = worse)
    """
    if len(expert_outputs) < 2:
        return torch.tensor(0.0, device=expert_outputs[0].device)

    # Mean-pool each expert's outputs to get a representative vector
    means = []
    for eo in expert_outputs:
        if eo.shape[0] > 0:
            means.append(eo.mean(dim=0))

    if len(means) < 2:
        return torch.tensor(0.0, device=expert_outputs[0].device)

    # Stack and normalize (float32 for numerical stability)
    means = torch.stack(means).float()  # (K, D)
    means_norm = F.normalize(means, dim=-1)  # (K, D)

    # Pairwise cosine similarity matrix
    sim = means_norm @ means_norm.T  # (K, K)

    # Penalize off-diagonal entries (squared)
    K = sim.shape[0]
    mask = ~torch.eye(K, dtype=torch.bool, device=sim.device)
    orth_loss = (sim[mask] ** 2).mean()

    return orth_loss


def compute_diversity_losses(
    expert_data_per_layer: list[dict],
    discriminator: ExpertDiscriminator,
) -> dict[str, Tensor]:
    """Compute orthogonality and discriminability losses with layer awareness.

    Orthogonality is computed *within each layer's expert set* and then averaged
    across layers, instead of pooling expert means across layers (which compares
    vectors from different residual streams and conflates `expert_id` across
    independent submodules).

    Discriminability is computed once over all expert outputs but with a
    per-token layer index passed to the discriminator, so the classifier can
    distinguish the same `expert_id` in different layers via the layer
    embedding.

    Args:
        expert_data_per_layer: list of aux dicts from MoELayer, each containing
            "expert_outputs" (list of tensors) and "expert_labels" (list of tensors)
        discriminator: ExpertDiscriminator module (constructed with
            num_layers=len(expert_data_per_layer) for the layer-conditioned head)

    Returns:
        dict with "orth_loss" and "disc_loss" tensors
    """
    device = expert_data_per_layer[0]["load_balance_loss"].device

    layer_orth_losses: list[Tensor] = []
    per_layer_disc: list[tuple[Tensor, Tensor, int]] = []  # (outputs, labels, layer_idx)

    for layer_idx, layer_data in enumerate(expert_data_per_layer):
        if "expert_outputs" not in layer_data:
            continue
        eos = layer_data["expert_outputs"]
        els = layer_data["expert_labels"]
        if not eos:
            continue

        # Per-layer orthogonality on this layer's expert mean vectors only.
        means = [eo.mean(dim=0) for eo in eos if eo.shape[0] > 0]
        if len(means) >= 2:
            means_t = torch.stack(means).float()  # (K, D)
            means_n = F.normalize(means_t, dim=-1)
            sim = means_n @ means_n.T  # (K, K)
            K = sim.shape[0]
            mask = ~torch.eye(K, dtype=torch.bool, device=sim.device)
            layer_orth_losses.append((sim[mask] ** 2).mean())

        per_layer_disc.append((torch.cat(eos, dim=0), torch.cat(els, dim=0), layer_idx))

    orth_loss = (
        torch.stack(layer_orth_losses).mean() if layer_orth_losses else torch.tensor(0.0, device=device)
    )

    if not per_layer_disc:
        return {"orth_loss": orth_loss, "disc_loss": torch.tensor(0.0, device=device)}

    all_outputs_cat = torch.cat([t[0] for t in per_layer_disc], dim=0)  # (M_total, D)
    all_labels_cat = torch.cat([t[1] for t in per_layer_disc], dim=0)  # (M_total,)
    layer_idx_per_token = torch.cat(
        [
            torch.full((t[0].shape[0],), t[2], dtype=torch.long, device=device)
            for t in per_layer_disc
        ],
        dim=0,
    )

    # Subsample if too many tokens to keep memory/compute reasonable
    max_disc_tokens = 4096
    if all_outputs_cat.shape[0] > max_disc_tokens:
        perm = torch.randperm(all_outputs_cat.shape[0], device=device)[:max_disc_tokens]
        all_outputs_cat = all_outputs_cat[perm]
        all_labels_cat = all_labels_cat[perm]
        layer_idx_per_token = layer_idx_per_token[perm]

    # Cast to float32 for discriminator (small MLP, float32 is fine)
    all_outputs_f32 = all_outputs_cat.float()

    # Cooperative single-CE loss: gradients flow to BOTH the discriminator
    # (which learns to classify which expert produced each output) and the
    # experts (which are pushed to be distinguishable in the discriminator's
    # feature space). The previous two-term formulation summed +ce(detach(x))
    # and -ce(x), which cancelled numerically (logged as 0) and gave the
    # discriminator a net-zero gradient — so it never trained.
    disc_logits = discriminator(all_outputs_f32, layer_idx_per_token)
    disc_loss = F.cross_entropy(disc_logits, all_labels_cat)

    return {"orth_loss": orth_loss, "disc_loss": disc_loss}
