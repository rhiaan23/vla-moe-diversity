"""Smoke test for the prefix-bottleneck pi0 path.

Constructs a small dummy batch, runs the training forward AND inference
sample_actions through the bottleneck path, and prints shapes + losses to
confirm wiring is sane before launching a real training job.

Run on a GPU node:
    python scripts/smoke_test_bottleneck.py
"""

from __future__ import annotations

import torch

from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import PI0Pytorch


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    cfg = PI0Config(
        device=device,
        dtype="bfloat16" if device == "cuda" else "float32",
        chunk_size=10,
        n_action_steps=10,
        num_inference_steps=2,
        max_state_dim=8,
        max_action_dim=8,
        empty_cameras=0,
        train_expert_only=True,
        freeze_vision_encoder=True,
        use_moe=True,
        moe_whole_expert=True,
        moe_num_experts=4,
        moe_top_k=2,
        lora_rank=8,
        lora_alpha=16.0,
        use_diversity_loss=True,
        moe_lambda_orth=0.05,
        prefix_bottleneck=True,
        prefix_bottleneck_dim=5,
        prefix_bottleneck_num_tokens=1,
        prefix_bottleneck_source="image_lang",
        tokenizer_max_length=8,
    )

    model = PI0Pytorch(cfg).to(device)
    if cfg.dtype == "bfloat16":
        # Match what to_bfloat16_for_selected_params does to the wrapped model.
        pass

    bsz = 2
    img_h = img_w = cfg.image_resolution[0]
    images = [torch.randn(bsz, 3, img_h, img_w, device=device, dtype=torch.float32)]
    img_masks = [torch.ones(bsz, dtype=torch.bool, device=device)]
    lang_tokens = torch.randint(0, 1000, (bsz, cfg.tokenizer_max_length), device=device)
    lang_masks = torch.ones(bsz, cfg.tokenizer_max_length, dtype=torch.bool, device=device)
    state = torch.randn(bsz, cfg.max_state_dim, device=device, dtype=torch.float32)
    actions = torch.randn(
        bsz, cfg.chunk_size, cfg.max_action_dim, device=device, dtype=torch.float32
    )

    print("Running training forward ...")
    losses, moe_loss_dict = model.forward(
        images=images,
        img_masks=img_masks,
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        state=state,
        actions=actions,
    )
    print(f"  losses.shape={tuple(losses.shape)}")
    print(f"  losses.mean={losses.mean().item():.4f}")
    print(f"  moe_loss_keys={list(moe_loss_dict.keys())}")
    print(f"  z (latest)={tuple(model._last_z.shape) if model._last_z is not None else None}")

    print("Running training backward ...")
    loss = losses.mean() + sum(
        v for k, v in moe_loss_dict.items() if isinstance(v, torch.Tensor) and v.requires_grad
    )
    loss.backward()
    proj_grad = model.prefix_bottleneck_projector.down[0].weight.grad
    print(f"  projector.down[0].grad nan-check: {torch.isnan(proj_grad).any().item()}")
    print(f"  projector.down[0].grad norm: {proj_grad.norm().item():.6f}")

    print("Running inference sample_actions ...")
    model.eval()
    with torch.no_grad():
        out = model.sample_actions(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
        )
    print(f"  sample_actions output shape={tuple(out.shape)}")
    print("All checks OK.")


if __name__ == "__main__":
    main()
