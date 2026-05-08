#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import builtins
import copy
import logging
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma

    from lerobot.policies.pi_gemma import (
        PaliGemmaForConditionalGenerationWithPiGemma,
        PiGemmaForCausalLM,
        _gated_residual,
        layernorm_forward,
    )
else:
    CONFIG_MAPPING = None
    modeling_gemma = None
    PiGemmaForCausalLM = None
    _gated_residual = None
    layernorm_forward = None
    PaliGemmaForConditionalGenerationWithPiGemma = None


from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi0.configuration_pi0 import DEFAULT_IMAGE_SIZE, PI0Config
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    OPENPI_ATTENTION_MASK_VALUE,
)


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    # Beta sampling uses _sample_dirichlet which isn't implemented for MPS, so sample on CPU
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)


def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else 0.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


class _MoEAdapter(nn.Module):
    """Drop-in adapter so `MoELayer` can replace a SwiGLU MLP inside the gemma expert.

    `MoELayer.forward()` returns `(tensor, aux_dict)`. Both `compute_layer_complete()`
    below and the standard HuggingFace Gemma decoder layer (used by the suffix-only
    inference path via `gemma_expert.model.forward(...)`) call `layer.mlp(x)` and
    expect a single tensor back. This adapter:

    - returns just the tensor from `forward()`,
    - stashes the latest aux dict on `self.last_aux` so the caller can collect it
      after each layer (mirrors how SmolVLA appends to `moe_aux_data`),
    - exposes the underlying first expert's `up_proj` so the dtype probe at
      `compute_layer_complete` (`if layer.mlp.up_proj.weight.dtype == ...`) keeps
      working without special-casing.
    """

    def __init__(self, moe_layer: nn.Module):
        super().__init__()
        self.moe_layer = moe_layer
        self.last_aux: dict | None = None
        self.collect_outputs: bool = False
        # In LoRA mode, pretrained Pi0 checkpoint keys (e.g. `...mlp.gate_proj.weight`)
        # need to be remapped to the MoE's shared frozen base FFN path
        # (`...mlp.moe_layer.base_mlp.gate_proj.weight`) at load time. Without this,
        # strict=False loading silently leaves base_mlp random-init.
        if getattr(moe_layer, "use_lora_experts", False):
            self._register_load_state_dict_pre_hook(self._remap_pretrained_mlp_keys)

    @staticmethod
    def _remap_pretrained_mlp_keys(
        state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            old_key = f"{prefix}{proj}.weight"
            new_key = f"{prefix}moe_layer.base_mlp.{proj}.weight"
            if old_key in state_dict and new_key not in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)

    @property
    def up_proj(self):
        # Delegate dtype-probe attribute access. In LoRA mode, experts carry LoRA
        # A/B only (no up_proj attr), so fall through to the shared base FFN.
        if getattr(self.moe_layer, "use_lora_experts", False):
            return self.moe_layer.base_mlp.up_proj
        return self.moe_layer.experts[0].up_proj

    def forward(self, x: Tensor) -> Tensor:
        out, aux = self.moe_layer(x, collect_expert_outputs=self.collect_outputs)
        self.last_aux = aux
        return out


class _WholeExpertMLPAdapter(nn.Module):
    """Tensor->tensor wrapper around `WholeExpertMoELayer`.

    Expert-routing decisions live on the wrapped module (set externally
    by the parent model before each forward), so this adapter only has to
    pass tensors through. Exposes `up_proj` like the original MLP so the
    dtype probe in `compute_layer_complete` keeps working.
    """

    def __init__(self, we_layer: nn.Module):
        super().__init__()
        self.we_layer = we_layer
        self._register_load_state_dict_pre_hook(self._remap_pretrained_mlp_keys)

    @staticmethod
    def _remap_pretrained_mlp_keys(
        state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # Pretrained checkpoint stores ...mlp.{gate,up,down}_proj.weight; remap to
        # the shared frozen base inside the WholeExpertMoELayer.
        for proj in ("gate_proj", "up_proj", "down_proj"):
            old_key = f"{prefix}{proj}.weight"
            new_key = f"{prefix}we_layer.base_mlp.{proj}.weight"
            if old_key in state_dict and new_key not in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)

    @property
    def up_proj(self):
        return self.we_layer.base_mlp.up_proj

    def forward(self, x: Tensor) -> Tensor:
        return self.we_layer(x)


class PrefixBottleneckProjector(nn.Module):
    """Compresses the pooled prefix to a low-dim ``z`` and synthesizes per-layer
    K/V tokens that the action expert cross-attends to in place of the real
    prefix. Net effect: the only prefix-side information any expert layer sees
    is a deterministic function of ``z``.

    - down MLP: pooled prefix (paligemma_width) -> hidden -> bottleneck_dim
    - per-layer up: z -> ``num_tokens * num_kv_heads * head_dim`` for K and V
      separately (one ``Linear`` pair per gemma_expert layer)
    """

    def __init__(
        self,
        prefix_dim: int,
        bottleneck_dim: int,
        hidden: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_tokens: int,
        state_dim: int = 0,
        down_num_layers: int = 2,
        upkv_num_layers: int = 1,
        upkv_hidden: int = 256,
        zero_init_upkv: bool = False,
    ) -> None:
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_tokens = num_tokens
        self.state_dim = state_dim
        kv_dim = num_kv_heads * head_dim
        in_dim = prefix_dim + state_dim
        down_layers: list[nn.Module] = []
        cur = in_dim
        for _ in range(max(down_num_layers - 1, 0)):
            down_layers.append(nn.Linear(cur, hidden))
            down_layers.append(nn.GELU())
            cur = hidden
        down_layers.append(nn.Linear(cur, bottleneck_dim))
        self.down = nn.Sequential(*down_layers)

        def _build_up_mlp() -> nn.Sequential:
            layers: list[nn.Module] = []
            cur = bottleneck_dim
            for _ in range(max(upkv_num_layers - 1, 0)):
                layers.append(nn.Linear(cur, upkv_hidden))
                layers.append(nn.GELU())
                cur = upkv_hidden
            final = nn.Linear(cur, num_tokens * kv_dim)
            if zero_init_upkv:
                nn.init.zeros_(final.weight)
                if final.bias is not None:
                    nn.init.zeros_(final.bias)
            layers.append(final)
            return nn.Sequential(*layers)

        self.up_k = nn.ModuleList([_build_up_mlp() for _ in range(num_layers)])
        self.up_v = nn.ModuleList([_build_up_mlp() for _ in range(num_layers)])

    def project_z(self, pooled_prefix: Tensor, state_emb: Tensor | None = None) -> Tensor:
        x = pooled_prefix.to(self.down[0].weight.dtype)
        if self.state_dim > 0:
            if state_emb is None:
                raise ValueError("state_emb is required when state_dim > 0 (prefix_bottleneck_include_state)")
            x = torch.cat([x, state_emb.to(x.dtype)], dim=-1)
        return self.down(x)

    def synth_layer_kv(self, z: Tensor, layer_idx: int) -> tuple[Tensor, Tensor]:
        """Return (K, V) of shape ``(B, num_kv_heads, num_tokens, head_dim)``."""
        bsz = z.shape[0]
        k = self.up_k[layer_idx](z).view(bsz, self.num_tokens, self.num_kv_heads, self.head_dim)
        v = self.up_v[layer_idx](z).view(bsz, self.num_tokens, self.num_kv_heads, self.head_dim)
        return k.transpose(1, 2), v.transpose(1, 2)


def compute_suffix_layer_with_synth_kv(
    layer_idx,
    suffix_embs,
    attention_mask,
    suffix_position_ids,
    adarms_cond,
    paligemma,
    gemma_expert,
    synth_k,
    synth_v,
):
    """Suffix-only layer forward with synthetic prefix K/V injected.

    ``synth_k``/``synth_v`` shape: ``(B, num_kv_heads, num_synth_tokens, head_dim)``.
    Synth tokens are placed at conceptual position 0; RoPE is applied only to
    the suffix slice of K (synth K is left un-rotated, which is equivalent to
    RoPE at position 0 being identity).
    """
    layer = gemma_expert.model.layers[layer_idx]
    suffix_hidden, gate = layernorm_forward(layer.input_layernorm, suffix_embs, adarms_cond)
    input_shape = suffix_hidden.shape[:-1]
    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
    q = layer.self_attn.q_proj(suffix_hidden).view(hidden_shape).transpose(1, 2)
    k = layer.self_attn.k_proj(suffix_hidden).view(hidden_shape).transpose(1, 2)
    v = layer.self_attn.v_proj(suffix_hidden).view(hidden_shape).transpose(1, 2)

    dummy = torch.zeros(
        q.shape[0], suffix_position_ids.shape[1], q.shape[-1], device=q.device, dtype=q.dtype
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy, suffix_position_ids)
    q, k = modeling_gemma.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

    full_k = torch.cat([synth_k.to(k.dtype), k], dim=2)
    full_v = torch.cat([synth_v.to(v.dtype), v], dim=2)

    scaling = paligemma.model.language_model.layers[layer_idx].self_attn.scaling
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.model.language_model.layers[layer_idx].self_attn,
        q,
        full_k,
        full_v,
        attention_mask,
        scaling,
    )
    head_dim = layer.self_attn.head_dim
    att_output = att_output.reshape(q.shape[0], -1, 1 * 8 * head_dim)

    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
    out_emb = layer.self_attn.o_proj(att_output)
    out_emb = _gated_residual(suffix_embs, out_emb, gate)
    after_first_residual = out_emb.clone()
    out_emb, gate2 = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond)
    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
        out_emb = out_emb.to(dtype=torch.bfloat16)
    out_emb = layer.mlp(out_emb)
    return _gated_residual(after_first_residual, out_emb, gate2)


# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(
    layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, paligemma, gemma_expert
):
    models = [paligemma.model.language_model, gemma_expert.model]
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond[i])
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    scaling = paligemma.model.language_model.layers[layer_idx].self_attn.scaling
    # Attention computation
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.model.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = paligemma.model.language_model.layers[layer_idx].self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = _gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()
        out_emb, gate = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond[i])
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = _gated_residual(after_first_residual, out_emb, gate)
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class GemmaConfig:  # see openpi `gemma.py: Config`
    """Configuration for Gemma model variants."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class PaliGemmaWithExpertModel(
    nn.Module
):  # see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
    """PaliGemma model with action expert for PI0."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        image_size: int = DEFAULT_IMAGE_SIZE,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
        use_moe: bool = False,
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        moe_expert_intermediate_size: int | None = 1024,
        use_diversity_loss: bool = False,
        use_lora_experts: bool = False,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
        moe_whole_expert: bool = False,
        moe_init_noise: float = 0.01,
        moe_whole_expert_use_sparse: bool = False,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.use_moe = use_moe
        self.use_diversity_loss = use_diversity_loss
        self.moe_whole_expert = moe_whole_expert
        self.moe_whole_expert_use_sparse = moe_whole_expert_use_sparse
        # Latest collected MoE auxiliary dicts (one per expert layer); refreshed
        # at the start of every forward() and read by PI0Pytorch.forward().
        self._last_moe_aux_data: list[dict] = []

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
        self.gemma_expert = PiGemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        # MoE: replace each expert layer's MLP with an MoE layer wrapped in an
        # adapter so it stays a tensor->tensor module from the caller's POV.
        # Mirrors `smolvlm_with_expert.py:131-144`.
        self.whole_expert_layers: list = []  # populated below if moe_whole_expert
        if use_moe and moe_whole_expert:
            from lerobot.policies.smolvla.moe import WholeExpertMoELayer

            expert_type = "sparse" if moe_whole_expert_use_sparse else "lora"
            for layer in self.gemma_expert.model.layers:
                we_layer = WholeExpertMoELayer(
                    base_mlp=layer.mlp,
                    num_experts=moe_num_experts,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    expert_type=expert_type,
                    init_noise=moe_init_noise,
                )
                layer.mlp = _WholeExpertMLPAdapter(we_layer)
                self.whole_expert_layers.append(we_layer)
        elif use_moe:
            from lerobot.policies.smolvla.moe import MoELayer

            # LoRA mode preserves the pretrained FFN by sharing a frozen reference
            # across all experts; intermediate-size shrink is mutually exclusive.
            effective_intermediate_size = None if use_lora_experts else moe_expert_intermediate_size

            for layer in self.gemma_expert.model.layers:
                moe_layer = MoELayer(
                    hidden_size=action_expert_config.width,
                    num_experts=moe_num_experts,
                    top_k=moe_top_k,
                    original_mlp=layer.mlp,
                    expert_intermediate_size=effective_intermediate_size,
                    use_lora_experts=use_lora_experts,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    init_noise=moe_init_noise,
                )
                layer.mlp = _MoEAdapter(moe_layer)

        self.to_bfloat16_for_selected_params(precision)
        self._set_requires_grad()

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # Keep full vision path in float32 so we never toggle (toggle causes optimizer
        # "same dtype" error). Align with PI05.
        params_to_keep_float32 = [
            "vision_tower",
            "multi_modal_projector",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
            for param in self.paligemma.model.vision_tower.parameters():
                param.requires_grad = False
        if self.train_expert_only:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
        if self.train_expert_only:
            self.paligemma.eval()

    def embed_image(self, image: torch.Tensor):
        # Vision tower and multi_modal_projector are kept in float32 (params_to_keep_float32). Align with PI05.
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        image_outputs = self.paligemma.model.get_image_features(image)
        features = image_outputs.pooler_output * self.paligemma.config.text_config.hidden_size**0.5
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.model.language_model.embed_tokens(tokens)

    def set_whole_expert_routing(self, indices: torch.Tensor, weights: torch.Tensor) -> None:
        """Broadcast a (B, k) per-sample expert assignment to every layer.

        Called by `PI0Pytorch` once per forward pass when whole-expert MoE is
        enabled, so the same expert index is used across all layers for a
        given sample.
        """
        for layer in self.whole_expert_layers:
            layer.set_routing(indices, weights)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]

        # Reset MoE auxiliary collection. The adapters write to .last_aux as a
        # side effect during the per-layer forward; we read them back below.
        # Mirrors how SmolVLA accumulates `moe_aux_data` (smolvlm_with_expert.py:443).
        self._last_moe_aux_data = []
        if self.use_moe:
            for layer in self.gemma_expert.model.layers:
                if isinstance(layer.mlp, _MoEAdapter):
                    layer.mlp.last_aux = None
                    layer.mlp.collect_outputs = self.use_diversity_loss

        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.model.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.model.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )

            # final norm
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = layernorm_forward(models[i].norm, hidden_states, adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

            # Collect MoE auxiliary outputs that the adapters stashed during
            # `compute_layer_complete`. Only the joint forward path needs this
            # (the prefix-only and suffix-only branches above are inference-time
            # codepaths and don't backprop the auxiliary losses).
            if self.use_moe:
                for layer in self.gemma_expert.model.layers:
                    if isinstance(layer.mlp, _MoEAdapter) and layer.mlp.last_aux is not None:
                        self._last_moe_aux_data.append(layer.mlp.last_aux)

        return [prefix_output, suffix_output], prefix_past_key_values

    def forward_with_bottleneck(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_2d_masks_4d: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        suffix_embs: torch.Tensor,
        suffix_attention_mask_4d: torch.Tensor,
        suffix_position_ids: torch.Tensor,
        adarms_cond,
        projector: "PrefixBottleneckProjector",
        prefix_source: str = "image_lang",
        lang_pad_mask: torch.Tensor | None = None,
        state_emb: torch.Tensor | None = None,
    ):
        """Bottleneck path: action expert sees only K/V tokens projected from a
        low-dim ``z``. Runs PaliGemma over prefix alone, pools the output, and
        runs gemma_expert over suffix alone with per-layer synthetic K/V.
        """
        if adarms_cond is None:
            adarms_cond = [None, None]

        # Reset MoE aux as in the standard forward.
        self._last_moe_aux_data = []
        if self.use_moe:
            for layer in self.gemma_expert.model.layers:
                if isinstance(layer.mlp, _MoEAdapter):
                    layer.mlp.last_aux = None
                    layer.mlp.collect_outputs = self.use_diversity_loss

        # 1) PaliGemma forward over prefix only. Force eager attention so the
        # additive 4D mask (float32 0/-inf) is accepted regardless of bf16 Q dtype.
        self.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        prefix_output = self.paligemma.model.language_model.forward(
            inputs_embeds=prefix_embs,
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            use_cache=False,
            adarms_cond=adarms_cond[0],
        ).last_hidden_state  # (B, P, paligemma_width)

        # 2) Pool over valid tokens. ``image_lang`` pools all valid prefix tokens;
        # ``lang_only`` pools only the language tokens.
        if prefix_source == "lang_only" and lang_pad_mask is not None:
            mask = (prefix_pad_masks & lang_pad_mask).to(prefix_output.dtype).unsqueeze(-1)
        else:
            mask = prefix_pad_masks.to(prefix_output.dtype).unsqueeze(-1)
        pooled = (prefix_output * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)

        # 3) Project to z (optionally including projected state).
        z = projector.project_z(pooled, state_emb=state_emb)  # (B, bottleneck_dim)

        # 4) Layer-by-layer suffix forward with per-layer synthetic K/V.
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        hidden_states = suffix_embs
        for layer_idx in range(num_layers):
            synth_k, synth_v = projector.synth_layer_kv(z, layer_idx)
            hidden_states = compute_suffix_layer_with_synth_kv(
                layer_idx=layer_idx,
                suffix_embs=hidden_states,
                attention_mask=suffix_attention_mask_4d,
                suffix_position_ids=suffix_position_ids,
                adarms_cond=adarms_cond[1],
                paligemma=self.paligemma,
                gemma_expert=self.gemma_expert,
                synth_k=synth_k,
                synth_v=synth_v,
            )

        # Final norm on suffix only.
        suffix_output, _ = layernorm_forward(
            self.gemma_expert.model.norm, hidden_states, adarms_cond[1]
        )

        # Collect MoE aux written by adapters during the per-layer pass.
        if self.use_moe:
            for layer in self.gemma_expert.model.layers:
                if isinstance(layer.mlp, _MoEAdapter) and layer.mlp.last_aux is not None:
                    self._last_moe_aux_data.append(layer.mlp.last_aux)

        return suffix_output, z


class PI0Pytorch(nn.Module):  # see openpi `PI0Pytorch`
    """Core PI0 PyTorch model."""

    def __init__(self, config: PI0Config, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, invalid resolution: {config.image_resolution}"
            )

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, False],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
            use_moe=config.use_moe,
            moe_num_experts=config.moe_num_experts,
            moe_top_k=config.moe_top_k,
            moe_expert_intermediate_size=config.moe_expert_intermediate_size,
            use_diversity_loss=config.use_diversity_loss,
            use_lora_experts=config.use_lora_experts,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            moe_whole_expert=config.moe_whole_expert,
            moe_init_noise=config.moe_init_noise,
            moe_whole_expert_use_sparse=config.moe_whole_expert_use_sparse,
        )

        # Diversity head: either the per-layer expert-output discriminator
        # (standard MoE-FFN mode) or the global whole-expert router (whole-
        # expert mode). Both live on PI0Pytorch so they aren't frozen by
        # `train_expert_only`. Mirrors `modeling_smolvla.py:588-597`.
        self.discriminator = None
        self.whole_expert_router = None
        self._last_router_aux: dict | None = None
        if config.use_moe and config.moe_whole_expert:
            from lerobot.policies.smolvla.moe import WholeExpertRouter

            # Router input: mean-pooled embedded prefix (image patches +
            # language tokens, both at paligemma_config.width) plus the
            # projected state vector (action_expert_config.width).
            router_in_dim = paligemma_config.width + action_expert_config.width
            self.whole_expert_router = WholeExpertRouter(
                input_dim=router_in_dim,
                num_experts=config.moe_num_experts,
                top_k=config.moe_top_k,
                hidden_size=config.moe_router_hidden_size,
                num_layers=config.moe_router_num_layers,
            )
        elif config.use_moe and config.use_diversity_loss:
            from lerobot.policies.smolvla.moe import ExpertDiscriminator

            self.discriminator = ExpertDiscriminator(
                hidden_size=action_expert_config.width,
                num_experts=config.moe_num_experts,
                num_layers=action_expert_config.depth,
                disc_hidden_size=config.moe_disc_hidden_size,
            )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)

        self.state_proj = nn.Linear(config.max_state_dim, action_expert_config.width)
        self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
        self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        # Prefix bottleneck: projector kept in float32 (small, numerically
        # sensitive). The synth K/V are cast to expert dtype at use time.
        self.prefix_bottleneck_projector: PrefixBottleneckProjector | None = None
        if config.prefix_bottleneck:
            if config.prefix_bottleneck_source not in ("image_lang", "lang_only"):
                raise ValueError(
                    f"Invalid prefix_bottleneck_source: {config.prefix_bottleneck_source}"
                )
            self.prefix_bottleneck_projector = PrefixBottleneckProjector(
                prefix_dim=paligemma_config.width,
                bottleneck_dim=config.prefix_bottleneck_dim,
                hidden=config.prefix_bottleneck_hidden,
                num_layers=action_expert_config.depth,
                num_kv_heads=action_expert_config.num_kv_heads,
                head_dim=action_expert_config.head_dim,
                num_tokens=config.prefix_bottleneck_num_tokens,
                state_dim=action_expert_config.width if config.prefix_bottleneck_include_state else 0,
                down_num_layers=config.prefix_bottleneck_num_layers,
                upkv_num_layers=config.prefix_bottleneck_upkv_num_layers,
                upkv_hidden=config.prefix_bottleneck_upkv_hidden,
                zero_init_upkv=config.prefix_bottleneck_zero_init_upkv,
            )
        self._last_z: Tensor | None = None

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            # Also compile the main forward pass used during training
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def _route_whole_experts(self, prefix_embs, prefix_pad_masks, state) -> None:
        """Run the global router and broadcast its decision to every layer.

        Router input: mean-pooled embedded prefix (image patches + language
        token embeddings, both produced by `embed_prefix` before the
        transformer runs) concatenated with the projected state vector. Pi0's
        vision tower is frozen, so the embedded image features are a
        sufficient "what's in the scene" signal without re-running the VLM.
        Stashes the router aux dict on `self._last_router_aux` for the loss
        aggregator to read.
        """
        if self.whole_expert_router is None:
            self._last_router_aux = None
            return
        # Mean-pool prefix tokens over the valid (pad-mask) positions.
        mask = prefix_pad_masks.float().unsqueeze(-1)
        pooled_prefix = (prefix_embs.float() * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        if self.state_proj.weight.dtype == torch.float32:
            state_in = state.to(torch.float32)
        else:
            state_in = state
        state_emb = self.state_proj(state_in).float()
        ctx = torch.cat([pooled_prefix, state_emb], dim=-1)
        idx, wts, aux = self.whole_expert_router(ctx)
        # Cast weights to expert dtype so per-sample dispatch math stays consistent.
        expert_dtype = next(self.paligemma_with_expert.gemma_expert.parameters()).dtype
        wts_cast = wts.to(expert_dtype)
        self.paligemma_with_expert.set_whole_expert_routing(idx, wts_cast)
        self._last_router_aux = aux

    def _aggregate_moe_losses(self, moe_aux_data: list[dict]) -> dict:
        """Aggregate per-layer MoE auxiliary outputs into scalar loss tensors.

        Mirrors the SmolVLA aggregation at `modeling_smolvla.py:826-841`. Returns
        a dict of named loss tensors (and util scalars). Caller is responsible
        for summing the trainable losses into the main loss.
        """
        moe_loss_dict: dict = {}
        if not self.config.use_moe:
            return moe_loss_dict

        # Whole-expert mode: aux comes from the global router (set by
        # _route_whole_experts) plus a parameter-space LoRA orth loss. Per-
        # layer aux from the FFN-MoE path is not produced.
        if self.config.moe_whole_expert:
            if self._last_router_aux is None:
                return moe_loss_dict
            aux = self._last_router_aux
            moe_loss_dict["moe_lb_loss"] = (
                aux["load_balance_loss"] * self.config.moe_load_balance_weight
            )
            moe_loss_dict["moe_expert_utilization_std"] = aux["tokens_per_expert"].std()

            # LoRA-mode-only diversity loss: probes ex.gate_A/gate_B which don't
            # exist on sparse experts. v5 (LoRA whole-expert) showed routing
            # specializes by task without this loss anyway, so for sparse we
            # rely on routing dynamics + sparse-upcycling init noise alone.
            if (
                self.config.use_diversity_loss
                and not self.config.moe_whole_expert_use_sparse
            ):
                from lerobot.policies.smolvla.moe import compute_lora_orthogonality_loss

                orth = compute_lora_orthogonality_loss(
                    self.paligemma_with_expert.whole_expert_layers,
                    n_probes=self.config.moe_lora_orth_probes,
                )
                moe_loss_dict["moe_orth_loss"] = orth * self.config.moe_lambda_orth
            return moe_loss_dict

        if not moe_aux_data:
            return moe_loss_dict

        lb_losses = [d["load_balance_loss"] for d in moe_aux_data]
        moe_loss_dict["moe_lb_loss"] = (
            torch.stack(lb_losses).mean() * self.config.moe_load_balance_weight
        )

        all_tokens_per_expert = torch.stack([d["tokens_per_expert"] for d in moe_aux_data]).mean(dim=0)
        moe_loss_dict["moe_expert_utilization_std"] = all_tokens_per_expert.std()

        task_router_ces = [d["task_router_ce"] for d in moe_aux_data if "task_router_ce" in d]
        if task_router_ces and self.config.moe_lambda_router_task_ce > 0.0:
            moe_loss_dict["moe_router_task_ce"] = (
                torch.stack(task_router_ces).mean() * self.config.moe_lambda_router_task_ce
            )

        if self.config.use_diversity_loss and self.discriminator is not None:
            from lerobot.policies.smolvla.moe import compute_diversity_losses

            diversity = compute_diversity_losses(moe_aux_data, self.discriminator)
            moe_loss_dict["moe_orth_loss"] = diversity["orth_loss"] * self.config.moe_lambda_orth
            moe_loss_dict["moe_disc_loss"] = diversity["disc_loss"] * self.config.moe_lambda_disc

        return moe_loss_dict

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def _embed_state(self, state: Tensor) -> Tensor:
        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        def state_proj_func(s):
            return self.state_proj(s)

        return self._apply_checkpoint(state_proj_func, state)

    def embed_suffix(self, state, noisy_actions, timestep, *, include_state_token: bool = True):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing.

        With ``include_state_token=False`` (option-b bottleneck), the state
        token is omitted from the suffix entirely — the action expert sees
        only ``[action+time]`` tokens. Caller is responsible for routing
        state info through ``project_z`` instead.
        """
        embs = []
        pad_masks = []
        att_masks = []
        device = state.device
        bsize = state.shape[0]

        if include_state_token:
            state_emb = self._embed_state(state)
            embs.append(state_emb[:, None, :])
            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=state_emb.device)
            pad_masks.append(state_mask)
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        def mlp_func(action_time_emb):
            x = self.action_time_mlp_in(action_time_emb)
            x = F.silu(x)
            return self.action_time_mlp_out(x)

        action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
        adarms_cond = None

        embs.append(action_time_emb)
        action_time_dim = action_time_emb.shape[1]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # State (when present) and first action both start a fresh attention block.
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def _build_bottleneck_masks(
        self,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        suffix_pad_masks: Tensor,
        suffix_att_masks: Tensor,
        num_synth_tokens: int,
    ):
        """Build the prefix-only and suffix-only attention masks/position_ids
        used by the bottleneck path. Synth tokens are appended at conceptual
        positions [0..T-1] with mask_ar=0 so suffix queries can attend to them
        regardless of suffix barriers.
        """
        bsize = prefix_pad_masks.shape[0]
        device = prefix_pad_masks.device

        # Prefix-only: standard self-attention over the full prefix.
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Synth + suffix combined for the suffix-only attention.
        synth_pad = torch.ones(bsize, num_synth_tokens, dtype=torch.bool, device=device)
        synth_att = torch.zeros(
            bsize, num_synth_tokens, dtype=suffix_att_masks.dtype, device=device
        )
        combined_pad = torch.cat([synth_pad, suffix_pad_masks], dim=1)
        combined_att = torch.cat([synth_att, suffix_att_masks], dim=1)
        combined_2d = make_att_2d_masks(combined_pad, combined_att)
        # Suffix queries attend over (synth | suffix) keys.
        suffix_attention_mask = combined_2d[:, num_synth_tokens:, :]
        combined_position_ids = torch.cumsum(combined_pad, dim=1) - 1
        suffix_position_ids = combined_position_ids[:, num_synth_tokens:]

        return (
            prefix_att_2d,
            prefix_position_ids,
            suffix_attention_mask,
            suffix_position_ids,
        )

    def _apply_policy_input_dropout(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        lang_tokens: Tensor,
        suffix_embs: Tensor,
        include_state_token: bool,
    ) -> tuple[Tensor, Tensor]:
        """Run-A input dropout: per-sample, per-modality independent zeroing.

        With probability ``policy_input_dropout``, zero out:
          * the state token (first row of the suffix, when present)
          * the image-patch positions of the prefix
          * the language-token positions of the prefix

        No-op outside training and when the dropout rate is 0.
        """
        p = self.config.policy_input_dropout
        if p <= 0.0 or not self.training:
            return prefix_embs, suffix_embs

        bsize = prefix_embs.shape[0]
        device = prefix_embs.device

        # Use a *separate* RNG so we don't advance the global CUDA RNG state.
        # The downstream model is wrapped with ``_apply_checkpoint`` using
        # ``preserve_rng_state=False``; if our torch.rand() calls advanced the
        # global state, any dropout/RNG inside the checkpointed forward would
        # be recomputed against a different RNG during backward, producing
        # mismatched activations and NaN gradients.
        gen = getattr(self, "_input_dropout_gen", None)
        if gen is None:
            gen = torch.Generator(device=device)
            # Seed once based on whatever the global RNG has, then never touch
            # the global state again from this generator.
            gen.manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))
            self._input_dropout_gen = gen

        # Independent Bernoulli rolls per sample per modality. Math runs in
        # float32 to avoid any bf16 numerical surprise; cast at multiply time.
        drop_state = torch.rand(bsize, device=device, generator=gen) < p
        drop_image = torch.rand(bsize, device=device, generator=gen) < p
        drop_lang = torch.rand(bsize, device=device, generator=gen) < p

        lang_mask = self._lang_pad_mask_in_prefix(prefix_pad_masks, lang_tokens)
        img_mask = prefix_pad_masks & ~lang_mask

        keep_per_pos = torch.ones_like(prefix_pad_masks, dtype=torch.float32)
        img_keep_b = (~drop_image).to(torch.float32)[:, None]  # (B, 1)
        lang_keep_b = (~drop_lang).to(torch.float32)[:, None]
        keep_per_pos = torch.where(
            img_mask, img_keep_b.expand_as(keep_per_pos), keep_per_pos
        )
        keep_per_pos = torch.where(
            lang_mask, lang_keep_b.expand_as(keep_per_pos), keep_per_pos
        )
        prefix_embs = prefix_embs * keep_per_pos.unsqueeze(-1).to(prefix_embs.dtype)

        if include_state_token:
            # State token is the first row of the suffix (see embed_suffix).
            # Build a (B, T, 1) per-token multiplier: state_keep at row 0, 1.0
            # elsewhere. Functional multiply avoids in-place ops that can
            # interact poorly with autograd / gradient checkpointing.
            seq_len = suffix_embs.shape[1]
            state_keep = (~drop_state).to(torch.float32)[:, None, None]  # (B, 1, 1)
            ones_rest = torch.ones(
                bsize, seq_len - 1, 1, dtype=torch.float32, device=device
            )
            state_factor = torch.cat([state_keep, ones_rest], dim=1)  # (B, T, 1)
            suffix_embs = suffix_embs * state_factor.to(suffix_embs.dtype)

        return prefix_embs, suffix_embs

    def _lang_pad_mask_in_prefix(
        self, prefix_pad_masks: Tensor, lang_tokens: Tensor
    ) -> Tensor:
        """Build a (B, P) mask True only at language-token positions of the prefix.

        ``embed_prefix`` always lays out images first, then language tokens, so
        the last ``lang_tokens.shape[1]`` columns of the prefix are language.
        AND with ``prefix_pad_masks`` to mask out language padding.
        """
        prefix_len = prefix_pad_masks.shape[1]
        num_lang = lang_tokens.shape[1]
        out = torch.zeros_like(prefix_pad_masks)
        out[:, prefix_len - num_lang :] = True
        return out & prefix_pad_masks

    def _forward_with_bottleneck(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        suffix_embs: Tensor,
        suffix_pad_masks: Tensor,
        suffix_att_masks: Tensor,
        adarms_cond,
        lang_tokens: Tensor,
        state: Tensor | None = None,
    ) -> Tensor:
        """Build masks then call ``PaliGemmaWithExpertModel.forward_with_bottleneck``."""
        T = self.config.prefix_bottleneck_num_tokens
        (
            prefix_att_2d,
            prefix_position_ids,
            suffix_attention_mask,
            suffix_position_ids,
        ) = self._build_bottleneck_masks(
            prefix_pad_masks, prefix_att_masks, suffix_pad_masks, suffix_att_masks, T
        )
        prefix_att_2d_4d = self._prepare_attention_masks_4d(prefix_att_2d)
        suffix_attention_mask_4d = self._prepare_attention_masks_4d(suffix_attention_mask)

        lang_pad_mask = None
        if self.config.prefix_bottleneck_source == "lang_only":
            lang_pad_mask = self._lang_pad_mask_in_prefix(prefix_pad_masks, lang_tokens)

        state_emb = None
        if self.config.prefix_bottleneck_include_state:
            if state is None:
                raise ValueError("state is required when prefix_bottleneck_include_state=True")
            state_emb = self._embed_state(state)

        suffix_out, z = self.paligemma_with_expert.forward_with_bottleneck(
            prefix_embs=prefix_embs,
            prefix_pad_masks=prefix_pad_masks,
            prefix_att_2d_masks_4d=prefix_att_2d_4d,
            prefix_position_ids=prefix_position_ids,
            suffix_embs=suffix_embs,
            suffix_attention_mask_4d=suffix_attention_mask_4d,
            suffix_position_ids=suffix_position_ids,
            adarms_cond=[None, adarms_cond],
            projector=self.prefix_bottleneck_projector,
            prefix_source=self.config.prefix_bottleneck_source,
            lang_pad_mask=lang_pad_mask,
            state_emb=state_emb,
        )
        self._last_z = z.detach()
        return suffix_out

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        # Option-b bottleneck: omit the state token from the suffix; state
        # info enters z via the projector instead. With
        # ``prefix_bottleneck_keep_state_token`` the state token is kept as a
        # residual pathway so the action expert sees state via both the
        # pretrained suffix self-attn and the synth K/V from z.
        include_state_token = (
            (not (self.config.prefix_bottleneck and self.config.prefix_bottleneck_include_state))
            or self.config.prefix_bottleneck_keep_state_token
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, time, include_state_token=include_state_token
        )

        # Run-A input dropout on (state token, image-prefix, lang-prefix).
        prefix_embs, suffix_embs = self._apply_policy_input_dropout(
            prefix_embs=prefix_embs,
            prefix_pad_masks=prefix_pad_masks,
            lang_tokens=lang_tokens,
            suffix_embs=suffix_embs,
            include_state_token=include_state_token,
        )

        # Whole-expert MoE: pick top-k experts per sample once for the whole
        # action-expert forward, so the same expert index is used at every layer.
        self._route_whole_experts(prefix_embs, prefix_pad_masks, state)

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        if self.config.prefix_bottleneck:
            suffix_out = self._forward_with_bottleneck(
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_att_masks=prefix_att_masks,
                suffix_embs=suffix_embs,
                suffix_pad_masks=suffix_pad_masks,
                suffix_att_masks=suffix_att_masks,
                adarms_cond=adarms_cond,
                lang_tokens=lang_tokens,
                state=state,
            )
        else:
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1

            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

            def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
                (_, suffix_out), _ = self.paligemma_with_expert.forward(
                    attention_mask=att_2d_masks_4d,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, suffix_embs],
                    use_cache=False,
                    adarms_cond=[None, adarms_cond],
                )
                return suffix_out

            suffix_out = self._apply_checkpoint(
                forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
            )

        # Collect MoE auxiliary outputs that PaliGemmaWithExpertModel.forward
        # stashed during the joint per-layer pass. Read here so the tensor refs
        # still belong to the live autograd graph for backprop.
        moe_aux_data = list(self.paligemma_with_expert._last_moe_aux_data)
        moe_loss_dict = self._aggregate_moe_losses(moe_aux_data)

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none"), moe_loss_dict

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = state.shape[0]
        device = state.device

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        bottleneck_active = (
            self.config.prefix_bottleneck and self.prefix_bottleneck_projector is not None
        )
        cached_synth_kv: list[tuple[Tensor, Tensor]] | None = None
        past_key_values = None

        if bottleneck_active:
            # Run paligemma over prefix once to compute the pooled feature, then
            # cache per-layer synth K/V from z so each denoise step is suffix-only.
            self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
            prefix_output = self.paligemma_with_expert.paligemma.model.language_model.forward(
                inputs_embeds=prefix_embs,
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                use_cache=False,
            ).last_hidden_state
            if self.config.prefix_bottleneck_source == "lang_only":
                lang_pad_mask = self._lang_pad_mask_in_prefix(prefix_pad_masks, lang_tokens)
                mask = (prefix_pad_masks & lang_pad_mask).to(prefix_output.dtype).unsqueeze(-1)
            else:
                mask = prefix_pad_masks.to(prefix_output.dtype).unsqueeze(-1)
            pooled = (prefix_output * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
            state_emb_for_z = (
                self._embed_state(state) if self.config.prefix_bottleneck_include_state else None
            )
            z = self.prefix_bottleneck_projector.project_z(pooled, state_emb=state_emb_for_z)
            self._last_z = z.detach()
            num_layers = self.paligemma_with_expert.paligemma.config.text_config.num_hidden_layers
            cached_synth_kv = [
                self.prefix_bottleneck_projector.synth_layer_kv(z, layer_idx)
                for layer_idx in range(num_layers)
            ]
        else:
            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

        # Whole-expert MoE: route once for the whole denoising loop. Same
        # expert assignment is reused across every diffusion step.
        self._route_whole_experts(prefix_embs, prefix_pad_masks, state)

        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                if bottleneck_active:
                    return self.denoise_step_bottleneck(
                        state=state,
                        prefix_pad_masks=prefix_pad_masks,
                        prefix_att_masks=prefix_att_masks,
                        cached_synth_kv=cached_synth_kv,
                        x_t=input_x_t,
                        timestep=current_timestep,
                    )
                return self.denoise_step(
                    state=state,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        past_key_values = copy.deepcopy(past_key_values)
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    def denoise_step_bottleneck(
        self,
        state,
        prefix_pad_masks,
        prefix_att_masks,
        cached_synth_kv,
        x_t,
        timestep,
    ):
        """One denoising step using cached per-layer synth K/V from the bottleneck.

        ``cached_synth_kv`` is a per-layer list of ``(K, V)`` tensors of shape
        ``(B, num_kv_heads, num_synth_tokens, head_dim)`` — pre-computed once
        per ``sample_actions`` call.
        """
        include_state_token = (
            not self.config.prefix_bottleneck_include_state
        ) or self.config.prefix_bottleneck_keep_state_token
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, timestep, include_state_token=include_state_token
        )

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0]
            .self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        T = self.config.prefix_bottleneck_num_tokens
        (
            _prefix_att_2d,
            _prefix_position_ids,
            suffix_attention_mask,
            suffix_position_ids,
        ) = self._build_bottleneck_masks(
            prefix_pad_masks, prefix_att_masks, suffix_pad_masks, suffix_att_masks, T
        )
        suffix_attention_mask_4d = self._prepare_attention_masks_4d(suffix_attention_mask)

        num_layers = self.paligemma_with_expert.paligemma.config.text_config.num_hidden_layers
        hidden_states = suffix_embs
        for layer_idx in range(num_layers):
            synth_k, synth_v = cached_synth_kv[layer_idx]
            hidden_states = compute_suffix_layer_with_synth_kv(
                layer_idx=layer_idx,
                suffix_embs=hidden_states,
                attention_mask=suffix_attention_mask_4d,
                suffix_position_ids=suffix_position_ids,
                adarms_cond=adarms_cond,
                paligemma=self.paligemma_with_expert.paligemma,
                gemma_expert=self.paligemma_with_expert.gemma_expert,
                synth_k=synth_k,
                synth_v=synth_v,
            )
        suffix_out, _ = layernorm_forward(
            self.paligemma_with_expert.gemma_expert.model.norm, hidden_states, adarms_cond
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class PI0Policy(PreTrainedPolicy):
    """PI0 OpenPI Policy for LeRobot."""

    config_class = PI0Config
    name = "pi0"

    def __init__(
        self,
        config: PI0Config,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize the core PI0 model
        self.init_rtc_processor()
        self.model = PI0Pytorch(config, rtc_processor=self.rtc_processor)

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Override the from_pretrained method to handle key remapping and display important disclaimer."""
        print(
            "The PI0 model is a direct port of the OpenPI implementation. \n"
            "This implementation follows the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        # Initialize model without loading weights
        # Check if dataset_stats were provided in kwargs
        model = cls(config, **kwargs)

        # Load state dict (expects keys with "model." prefix)
        try:
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    token=kwargs.get("token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                print(f"Could not load state dict from remote files: {e}")
                print("Returning model without loading pretrained weights")
                return model

            # First, fix any key differences (see openpi model.py, _fix_pytorch_state_dict_keys)
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            # Load the remapped state dict into the model
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=strict)

            # Free intermediate state dicts to recover ~32GB of CPU RAM
            del original_state_dict, fixed_state_dict, remapped_state_dict
            import gc

            gc.collect()

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        except Exception as e:
            print(f"Warning: Could not load state dict: {e}")

        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi0
            # non-pi05 model expects action_time_mlp_*, but checkpoint might have time_mlp_*
            if key.startswith("time_mlp_in."):
                new_key = key.replace("time_mlp_in.", "action_time_mlp_in.")
            elif key.startswith("time_mlp_out."):
                new_key = key.replace("time_mlp_out.", "action_time_mlp_out.")

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            if (
                key == "model.paligemma_with_expert.paligemma.lm_head.weight"
                or key == "paligemma_with_expert.paligemma.lm_head.weight"
            ):
                fixed_state_dict[
                    "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
                ] = value.clone()

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []

        # Get device from model parameters
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        for key in present_img_keys:
            img = batch[key]

            # Ensure tensor is on the same device as the model
            if img.device != device:
                img = img.to(device)

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)

            # from openpi preprocess_observation_pytorch: Resize with padding if needed
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_state(self, batch):
        """Pad state"""
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        lang_tokens, lang_masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        state = self.prepare_state(batch)

        # Sample actions using the model (pass through RTC kwargs)
        actions = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, **kwargs)

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Args:
            batch: Training batch containing observations and actions.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        lang_tokens, lang_masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)

        # Make per-batch task_index visible to MoELayer.forward via contextvar.
        # Used only when moe_lambda_router_task_ce > 0 (see moe.py).
        from lerobot.policies.smolvla.moe import set_current_task_index

        task_idx = batch.get("task_index", None)
        set_current_task_index(task_idx)
        try:
            losses, moe_loss_dict = self.model.forward(
                images, img_masks, lang_tokens, lang_masks, state, actions
            )
        finally:
            set_current_task_index(None)

        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }

        # Sum MoE auxiliary losses (tensors with grad) for backprop, log scalars
        # for everything in moe_loss_dict. Mirrors `modeling_smolvla.py:395-411`.
        moe_aux_total = torch.tensor(0.0, device=losses.device)
        for k, v in moe_loss_dict.items():
            if isinstance(v, torch.Tensor) and v.requires_grad:
                moe_aux_total = moe_aux_total + v
            loss_dict[k] = v.item() if isinstance(v, torch.Tensor) else v

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_loss = losses.mean(dim=(1, 2))
            total = per_sample_loss.mean() + moe_aux_total
            loss_dict["loss"] = total.item()
            # Spread the (scalar) aux total evenly across the batch dim so the
            # caller's reduction still gives the right value.
            return per_sample_loss + moe_aux_total / per_sample_loss.shape[0], loss_dict
        else:
            # Default: return scalar mean loss
            loss = losses.mean() + moe_aux_total
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for PI0 fine-tuning.

        Two regimes:

        * ``lora_vlm=False`` (default): LoRA adapters target the action-expert
          q/v projections and the small I/O heads. The base VLM is kept fully
          frozen.
        * ``lora_vlm=True``: LoRA adapters target the *PaliGemma VLM* text-
          model q/v projections, and the whole-expert router + I/O heads are
          listed under ``modules_to_save`` so they are trained full-rank.
          The gemma_expert action transformer (self-attn, layer norms, and
          the per-layer LoRA experts inside ``WholeExpertMoELayer``) stays
          frozen — this is the "experts frozen, finetune router + VLM-LoRA"
          recipe used to recompose pretrained MoE skills onto new tasks.
        """
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        if self.config.lora_vlm:
            target_modules = (
                r".*\.paligemma\.model\.language_model\.layers\.\d+\.self_attn\.(q|v)_proj"
            )
            modules_to_save: list[str] = [
                "state_proj",
                "action_in_proj",
                "action_out_proj",
                "action_time_mlp_in",
                "action_time_mlp_out",
            ]
            if self.config.use_moe and self.config.moe_whole_expert:
                modules_to_save.append("whole_expert_router")
                # Whole-expert LoRA experts live inside ``WholeExpertMoELayer``
                # (named ``we_layer`` on each gemma_expert layer). Without
                # adding them to ``modules_to_save``, PEFT freezes them and
                # they stay at their random init.
                modules_to_save.append("we_layer")
            elif self.config.use_moe and self.config.use_diversity_loss:
                modules_to_save.append("discriminator")
            # Prefix bottleneck projector is added by us; PEFT freezes it by
            # default since it's neither a target adapter nor in the original
            # modules_to_save list.
            if self.config.prefix_bottleneck:
                modules_to_save.append("prefix_bottleneck_projector")
            return {
                "target_modules": target_modules,
                "modules_to_save": modules_to_save,
            }
        target_modules = rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }
