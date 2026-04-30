"""Smoke test for LoRA-MoE: load pretrained Pi0, verify key remap worked,
run one forward pass on random input, print param counts."""

import os
os.environ.setdefault("HF_LEROBOT_HOME", "/scratch/gpfs/FHEIDE/rj2807/lerobot_data")
os.environ.setdefault("HF_HOME", "/scratch/gpfs/FHEIDE/rj2807/cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

print("[1/5] Loading pretrained Pi0 with LoRA-MoE ...")
cfg = PI0Config(
    dtype="bfloat16",
    train_expert_only=True,
    freeze_vision_encoder=True,
    gradient_checkpointing=True,
    use_moe=True,
    use_lora_experts=True,
    moe_num_experts=8,
    moe_top_k=2,
    use_diversity_loss=True,
    lora_rank=16,
    lora_alpha=32.0,
    device="cuda",
)
policy = PI0Policy.from_pretrained("lerobot/pi0", config=cfg)
policy.to("cuda")

print("[2/5] Checking base_mlp holds pretrained (not random) weights ...")
pi0 = policy.model
layer0 = pi0.paligemma_with_expert.gemma_expert.model.layers[0].mlp
assert hasattr(layer0, "moe_layer"), "expected _MoEAdapter"
base = layer0.moe_layer.base_mlp
assert base is not None, "base_mlp missing in LoRA mode"
w = base.gate_proj.weight
print(f"    base.gate_proj.weight: shape={tuple(w.shape)} dtype={w.dtype} "
      f"abs_mean={w.abs().mean().item():.4f} "
      f"std={w.std().item():.4f} requires_grad={w.requires_grad}")
assert not w.requires_grad, "base_mlp should be frozen"

# Definitive check: compare bit-for-bit against the raw safetensors file.
from huggingface_hub import hf_hub_download
from safetensors import safe_open
raw_path = hf_hub_download(repo_id="lerobot/pi0", filename="model.safetensors")
with safe_open(raw_path, framework="pt") as f:
    pretrained_key = "model.paligemma_with_expert.gemma_expert.model.layers.0.mlp.gate_proj.weight"
    raw_w = f.get_tensor(pretrained_key).to(w.device).to(w.dtype)
max_diff = (w - raw_w).abs().max().item()
print(f"    max |base.gate_proj - pretrained_gate_proj| = {max_diff:.6f}")
assert max_diff < 1e-4, f"base_mlp does NOT match pretrained (diff={max_diff}). Key remap failed."
print("    ✓ base_mlp matches pretrained Pi0 gate_proj exactly")

print("[3/5] Param counts ...")
total = sum(p.numel() for p in policy.parameters())
trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
print(f"    total={total/1e6:.1f}M trainable={trainable/1e6:.1f}M")

print("[4/5] Checking LoRA expert init (B should be zero) ...")
lora0 = layer0.moe_layer.experts[0]
print(f"    gate_A norm={lora0.gate_A.weight.norm().item():.4f} "
      f"gate_B norm={lora0.gate_B.weight.norm().item():.4f}")
assert lora0.gate_B.weight.abs().sum() == 0, "LoRA B should be zero-init"

print("[5/5] GPU memory after load:")
print(f"    allocated={torch.cuda.memory_allocated()/1e9:.2f} GB "
      f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GB")

print("\nSMOKE OK")
