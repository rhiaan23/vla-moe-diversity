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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("pi0")
@dataclass
class PI0Config(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10  # Number of denoising steps during inference
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Relative actions: converts absolute actions to relative (relative to state).
    use_relative_actions: bool = False
    # Joint names to exclude from relative (kept absolute). Empty list = all dims relative.
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # Populated at runtime from dataset metadata by make_policy.
    action_feature_names: list[str] | None = None

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Finetuning settings
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder
    train_expert_only: bool = False  # Freeze entire VLM, train only action expert and projections

    # Optimizer settings: see openpi `AdamW``
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 48  # see openpi `__post_init__`

    # MoE configuration (mirrors SmolVLAConfig). The action expert's per-layer
    # SwiGLU MLP is replaced with `MoELayer(num_experts, top_k)` from
    # `lerobot.policies.smolvla.moe`. Defaults match the SmolVLA matrix.
    use_moe: bool = False
    moe_num_experts: int = 8
    moe_top_k: int = 2
    # gemma_300m action expert has mlp_dim=4096; 1024 keeps the per-expert SwiGLU
    # roughly 1/4 the size of the original MLP so 8 experts ≈ 2× the params of
    # the baseline MLP, comparable to the SmolVLA setup.
    # Only used when use_lora_experts=false AND moe_expert_intermediate_size is not None.
    moe_expert_intermediate_size: int | None = 1024
    moe_load_balance_weight: float = 0.01
    # Std of Gaussian noise added to each expert's parameters at sparse-upcycling
    # init (see moe.MoELayer). 0.01 keeps experts near-clones; 0.1 breaks
    # symmetry harder so the discriminator + router actually have something to
    # latch onto early in training.
    moe_init_noise: float = 0.01

    # LoRA-MoE: each expert = frozen pretrained FFN + per-expert LoRA delta.
    # When true, ignores moe_expert_intermediate_size and preserves pretrained init.
    use_lora_experts: bool = False
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0

    # Whole-expert MoE: N LoRA-adapted action experts share a frozen base FFN,
    # but the SAME expert index is chosen for a given sample across ALL layers
    # (sample-level routing). Implies LoRA experts. Diversity loss switches to
    # functional orthogonality on LoRA delta directions instead of the
    # per-layer expert-output discriminator.
    moe_whole_expert: bool = False
    # When moe_whole_expert=true, choose the per-expert capacity:
    #   false → "lora" experts (frozen base + per-expert LoRA delta)
    #   true  → "sparse" experts (deep-copies of pretrained MLP + init noise)
    # Sparse mode combines v5's sample-level routing (proven task-conditional)
    # with v3's full-FFN capacity (proven to retain task success).
    moe_whole_expert_use_sparse: bool = False
    # Router input source. "prefix_state" = mean-pooled embedded prefix
    # (image patches + language tokens) concatenated with the projected
    # state vector — the cheapest signal that includes vision.
    moe_router_input: str = "prefix_state"
    # Router architecture (deeper than a single linear so it can learn
    # non-linear skill boundaries).
    moe_router_hidden_size: int = 256
    moe_router_num_layers: int = 3
    # Number of random probe vectors used by the LoRA orthogonality loss.
    moe_lora_orth_probes: int = 64

    # Diversity losses (orthogonality + discriminability) on top of standard MoE.
    use_diversity_loss: bool = False
    moe_lambda_orth: float = 0.05
    moe_lambda_disc: float = 0.02
    moe_disc_hidden_size: int = 128

    # Task-supervised router CE: forces task-conditional routing by training
    # the router to map task_id mod num_experts to the chosen expert. Empirical
    # evidence (v6/v7) shows pure-BC diversity loss cannot drive task-conditional
    # routing on its own; this provides the missing supervision signal.
    moe_lambda_router_task_ce: float = 0.0

    # LoRA fine-tuning of the (otherwise frozen) PaliGemma VLM. When True, the
    # default PEFT targets are switched so that LoRA adapters are applied to
    # the VLM text-model self-attention q/v projections, while the
    # whole-expert router and the I/O heads (state_proj, action_*_proj,
    # action_time_mlp_*) are placed in ``modules_to_save`` (i.e., trained
    # full-rank). The action expert (gemma_expert) — including its self-
    # attention, layernorms, and the per-layer LoRA experts inside
    # ``WholeExpertMoELayer`` — stays frozen. Together this yields the
    # "freeze experts, finetune router + VLM-LoRA" recipe.
    # NOTE: this flag only takes effect if PEFT is also enabled, e.g. by
    # passing ``--peft.method_type=LORA`` on the command line. Without that,
    # the policy trains in the standard non-PEFT way.
    lora_vlm: bool = False

    # Post-load freezing knobs for the "router-only finetune" experiment, where
    # the v5 LoRA experts are treated as fixed primitives and only the router
    # is allowed to update. Both freeze passes run after make_policy + (any)
    # PEFT wrap, so they take effect with or without PEFT.
    # ``freeze_action_expert`` freezes the entire gemma_expert subtree
    # (self-attn, layernorms, *and* every WholeExpertMoELayer + its LoRA
    # experts), plus the discriminator if present.
    freeze_action_expert: bool = False
    # ``freeze_io_heads`` freezes state_proj, action_in_proj, action_out_proj,
    # action_time_mlp_in, action_time_mlp_out so the only thing that moves is
    # the whole_expert_router (assuming PaliGemma is already frozen via
    # train_expert_only=True).
    freeze_io_heads: bool = False

    # Prefix bottleneck: replace the per-layer prefix K/V the action expert
    # cross-attends to with a single (or few) synthetic tokens whose K/V are
    # produced from a low-dimensional projection of the pooled prefix. The
    # action expert ends up "seeing" only this low-dim control vector instead
    # of the full image+language sequence.
    prefix_bottleneck: bool = False
    prefix_bottleneck_dim: int = 5
    prefix_bottleneck_num_tokens: int = 1
    # Source of the pooled prefix that feeds the bottleneck:
    #   "image_lang" → mean-pool over PaliGemma prefix tokens (image+language)
    #   "lang_only"  → mean-pool over language tokens only
    prefix_bottleneck_source: str = "image_lang"
    prefix_bottleneck_hidden: int = 256
    # Depth of the down projector (Linear count). Default 2 matches the
    # original (Linear → GELU → Linear) architecture; bump to 3+ to deepen
    # the projector (e.g. to match a wider/deeper router).
    prefix_bottleneck_num_layers: int = 2
    # Option (b): also bottleneck proprioceptive state. When True, the state
    # is concatenated into the projector's input (so z encodes vision + lang
    # + state) and the state token is REMOVED from the suffix — the action
    # expert literally only sees actions+time as inputs and the 5-D z + chosen
    # router idx as conditioning. Test of "expert as fully open-loop skill".
    prefix_bottleneck_include_state: bool = False
    # Residual state pathway: when ``include_state=True`` the state token is
    # normally removed from the suffix (option-b). Setting this True keeps
    # the state token in the suffix as a residual, so the action expert sees
    # state via two channels — the pretrained suffix self-attention (calibrated
    # at init) AND the synth K/V via z. Combined with ``zero_init_upkv``, the
    # bottleneck path is silent at init and the policy is identical to the
    # pretrained Pi0-with-state-token at step 0.
    prefix_bottleneck_keep_state_token: bool = False
    # Zero-initialise the final layer of every per-layer up_k / up_v MLP so
    # that synth K/V ≡ 0 at init. This pairs with ``keep_state_token`` to give
    # the policy a working starting point: residual state token feeds the
    # pretrained pathway, bottleneck path contributes nothing until up_k/up_v
    # learn nonzero values during training.
    prefix_bottleneck_zero_init_upkv: bool = False
    # Capacity of the per-layer up_k / up_v MLPs that synthesize the K/V the
    # action expert reads in place of the real prefix. Default 1 keeps the
    # original single-Linear behaviour. Set higher (e.g. 4) so the synth path
    # can learn the calibrated K/V distribution the pretrained action expert
    # was trained against — capacity here is the actual binding constraint
    # in option-b, not the down-projector's bandwidth.
    prefix_bottleneck_upkv_num_layers: int = 1
    prefix_bottleneck_upkv_hidden: int = 256

    # Policy input dropout (Run A: regularise the policy directly without a
    # bottleneck). At training time, with probability ``policy_input_dropout``
    # *per modality, per sample, independent rolls*, zero out:
    #   * the state token's embedding in the suffix
    #   * the image-patch positions in the prefix output (before cross-attn)
    #   * the language-token positions in the prefix output
    # Goal: experts learn to be robust to missing modalities → behave more
    # like generic primitives without the broken bootstrap of a learned
    # bottleneck. Has no effect at eval time. 0.0 disables.
    policy_input_dropout: float = 0.0

    # Deprecated fields kept for backward compatibility with pretrained checkpoint configs.
    # These existed in earlier versions of the Pi0 config and are present in saved YAML files
    # (e.g. lerobot/pi0_base). They are unused by current code.
    resize_imgs_with_padding: list[int] | None = None
    adapt_to_pi_aloha: bool = False
    use_delta_joint_actions_aloha: bool = False
    proj_width: int | None = None
    num_steps: int | None = None
    use_cache: bool = False
    attention_implementation: str | None = None
    train_state_proj: bool = False

    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
