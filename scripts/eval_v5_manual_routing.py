"""Manual-routing eval for v5 (whole-expert MoE).

Replaces ``WholeExpertRouter.forward`` with a closure that returns per-chunk
expert assignments from a JSON program (scripted), stdin (interactive),
random sampling (control), or the original router (passthrough).

Mirrors the monkey-patch / atexit-dump pattern in ``eval_v5_routing.py`` so
``routing_log.json`` keeps the same shape (with extra fields for the manual
choice vs the router's "would-have" choice).

Extra CLI flags (peeled off ``sys.argv`` before draccus parses the rest):
  --manual_mode={scripted,interactive,random,passthrough}
  --manual_program=<path.json>      (required for scripted)
  --manual_random_seed=<int>        (optional for random; default = wall time)

Standard lerobot_eval flags pass through unchanged. ``--output_dir`` is read
the same way ``eval_v5_routing.py`` reads it.

Program JSON schema (scripted):
  {
    "task": "libero_10/T6",            # informational only
    "by_chunk": [                      # one entry per chunk index
      {"indices": [11], "weights": [1.0]},
      {"indices": [11, 5], "weights": [0.5, 0.5]}
    ],
    "default": {"indices": [11, 5], "weights": [0.7, 0.3]}
  }

Per-chunk lookup falls back to ``default`` once ``chunk_idx >=
len(by_chunk)``. Indices and weights are padded/truncated to the policy's
configured ``top_k`` (zero-weight pads are skipped by
``WholeExpertMoELayer.forward`` so they have no effect).
"""

from __future__ import annotations

import atexit
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

import lerobot.scripts.lerobot_eval as lev


# ---------------------------------------------------------------------------
# CLI helpers (peel custom flags before draccus parses)
# ---------------------------------------------------------------------------

def _peel_flag(argv: list[str], key: str) -> str | None:
    """Pop ``--key=value`` (or ``--key value``) from argv, return the value."""
    prefix = f"--{key}="
    for i, a in enumerate(argv):
        if a.startswith(prefix):
            val = a.split("=", 1)[1]
            argv.pop(i)
            return val
        if a == f"--{key}" and i + 1 < len(argv):
            val = argv[i + 1]
            argv.pop(i + 1)
            argv.pop(i)
            return val
    return None


def _read_arg_no_pop(argv: list[str], key: str) -> str | None:
    prefix = f"--{key}="
    for i, a in enumerate(argv):
        if a.startswith(prefix):
            return a.split("=", 1)[1]
        if a == f"--{key}" and i + 1 < len(argv):
            return argv[i + 1]
    return None


MANUAL_MODE = _peel_flag(sys.argv, "manual_mode") or "scripted"
MANUAL_PROGRAM_PATH = _peel_flag(sys.argv, "manual_program")
MANUAL_RANDOM_SEED = _peel_flag(sys.argv, "manual_random_seed")
OUTPUT_DIR = _read_arg_no_pop(sys.argv, "output_dir")
OUTPUT_DIR_PATH = Path(OUTPUT_DIR) if OUTPUT_DIR else None

if MANUAL_MODE not in ("scripted", "interactive", "random", "passthrough"):
    raise SystemExit(f"unknown --manual_mode={MANUAL_MODE}")

if MANUAL_MODE == "scripted":
    if not MANUAL_PROGRAM_PATH:
        raise SystemExit("--manual_program=<path.json> is required for scripted mode")
    PROGRAM = json.load(open(MANUAL_PROGRAM_PATH))
elif MANUAL_MODE == "interactive":
    PROGRAM = None
    # Force single-env single-episode for sanity. The user can still pass
    # --eval.batch_size=1 and --eval.n_episodes=1 explicitly; we just warn.
    if _read_arg_no_pop(sys.argv, "eval.batch_size") not in (None, "1"):
        print("[manual_routing] WARNING: interactive mode strongly assumes --eval.batch_size=1")
elif MANUAL_MODE == "random":
    PROGRAM = None
    seed = int(MANUAL_RANDOM_SEED) if MANUAL_RANDOM_SEED is not None else int(time.time()) & 0xFFFFFFFF
    random.seed(seed)
    print(f"[manual_routing] random mode seed={seed}")
else:  # passthrough
    PROGRAM = None


# ---------------------------------------------------------------------------
# Manual router
# ---------------------------------------------------------------------------


class ManualRouter:
    def __init__(
        self,
        policy,
        mode: str,
        program: dict | None,
        top_k: int,
        num_experts: int,
        image_dump_dir: Path | None,
    ):
        self.policy = policy
        self.mode = mode
        self.program = program
        self.top_k = top_k
        self.num_experts = num_experts
        self.image_dump_dir = image_dump_dir

        self.records: list[dict] = []
        self.chunk_idx = 0
        self.episode_counter = 0  # increments per policy.reset (per parallel batch)
        self.current_task: tuple[str, int] = ("?", -1)
        self.captured_obs: dict | None = None

        # Save originals
        self._orig_router_forward = policy.model.whole_expert_router.forward
        self._orig_predict_chunk = policy.predict_action_chunk
        self._orig_reset = policy.reset

    def install(self) -> None:
        policy = self.policy

        def _patched_predict(batch, **kwargs):
            self.captured_obs = batch
            return self._orig_predict_chunk(batch, **kwargs)

        def _patched_reset(*args, **kwargs):
            self.chunk_idx = 0
            self.episode_counter += 1
            return self._orig_reset(*args, **kwargs)

        policy.predict_action_chunk = _patched_predict
        policy.reset = _patched_reset
        policy.model.whole_expert_router.forward = self._manual_forward

    def set_task(self, group: str, task_id: int) -> None:
        self.current_task = (group, int(task_id))
        self.episode_counter = 0  # reset per task

    # -- per-chunk dispatch --------------------------------------------------

    def _manual_forward(self, ctx: torch.Tensor):
        # ctx: (B, D). Always run the original router so we capture the aux
        # dict and what the router *would* have picked for the log.
        idx_orig, wts_orig, aux = self._orig_router_forward(ctx)
        B, k = idx_orig.shape
        device = idx_orig.device

        if self.mode == "scripted":
            choice = self._lookup_scripted()
        elif self.mode == "interactive":
            choice = self._prompt_interactive(idx_orig, wts_orig)
        elif self.mode == "random":
            choice = self._random_choice()
        else:  # passthrough
            choice = None

        if choice is None:
            new_idx, new_wts = idx_orig, wts_orig
        else:
            new_idx, new_wts = self._broadcast_choice(choice, B, k, device, idx_orig.dtype, wts_orig.dtype)

        self.records.append(
            {
                "group": self.current_task[0],
                "task": self.current_task[1],
                "episode_batch": self.episode_counter,
                "chunk_idx": self.chunk_idx,
                "mode": self.mode,
                "manual_indices": new_idx.detach().cpu().tolist(),
                "manual_weights": new_wts.detach().to(torch.float32).cpu().tolist(),
                "router_indices": idx_orig.detach().cpu().tolist(),
                "router_weights": wts_orig.detach().to(torch.float32).cpu().tolist(),
            }
        )
        self.chunk_idx += 1
        return new_idx, new_wts, aux

    # -- mode implementations ------------------------------------------------

    def _lookup_scripted(self) -> dict:
        by_chunk = self.program.get("by_chunk", [])
        if isinstance(by_chunk, list) and self.chunk_idx < len(by_chunk):
            return by_chunk[self.chunk_idx]
        default = self.program.get("default")
        if default is None:
            raise RuntimeError(
                f"scripted program ran out of by_chunk entries at chunk_idx={self.chunk_idx} "
                f"and no 'default' is set"
            )
        return default

    def _random_choice(self) -> dict:
        k = self.top_k
        # Use builtin random (seeded above) so runs are reproducible.
        indices = random.sample(range(self.num_experts), k)
        return {"indices": indices, "weights": [1.0 / k] * k}

    def _prompt_interactive(self, idx_orig: torch.Tensor, wts_orig: torch.Tensor) -> dict | None:
        # Dump the most recent obs frame so the user can look at it.
        if self.image_dump_dir is not None and self.captured_obs is not None:
            try:
                self._dump_frames(self.captured_obs, self.chunk_idx)
            except Exception as e:
                print(f"[manual_routing] image dump failed: {e}")

        print("\n" + "=" * 70)
        print(f"[chunk {self.chunk_idx}] task={self.current_task} batch_episode={self.episode_counter}")
        if self.captured_obs is not None:
            for k in self.captured_obs:
                if "task" in k.lower() and "image" not in k.lower():
                    val = self.captured_obs[k]
                    if hasattr(val, "tolist"):
                        try:
                            val = val.tolist()
                        except Exception:
                            pass
                    print(f"  {k}: {val}")
        print(
            f"  router would pick: indices={idx_orig[0].cpu().tolist()} "
            f"weights={[round(w, 3) for w in wts_orig[0].float().cpu().tolist()]}"
        )
        if self.image_dump_dir is not None:
            print(f"  obs frames -> {self.image_dump_dir}/chunk_{self.chunk_idx:03d}_*.png")
        prompt = (
            f"  enter top-{self.top_k} (e.g. '11' or '11,5' or '11:0.7,5:0.3', "
            f"empty=passthrough, q=quit): "
        )
        try:
            s = input(prompt).strip()
        except EOFError:
            print("[manual_routing] EOF on stdin, falling back to passthrough for the rest")
            self.mode = "passthrough"
            return None
        if s in ("q", "quit", "exit"):
            raise SystemExit("user quit interactive mode")
        if not s:
            return None
        return self._parse_input(s)

    @staticmethod
    def _parse_input(s: str) -> dict:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        indices: list[int] = []
        weights: list[float] = []
        for p in parts:
            if ":" in p:
                e, w = p.split(":", 1)
                indices.append(int(e))
                weights.append(float(w))
            else:
                indices.append(int(p))
                weights.append(1.0)
        return {"indices": indices, "weights": weights}

    # -- choice -> tensor ----------------------------------------------------

    def _broadcast_choice(
        self,
        choice: dict,
        B: int,
        k: int,
        device: torch.device,
        idx_dtype: torch.dtype,
        wts_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = list(choice["indices"])
        weights = list(choice.get("weights", [1.0] * len(indices)))
        if len(weights) != len(indices):
            raise ValueError(f"choice indices/weights length mismatch: {choice}")
        # Pad to k with zero-weight slot 0 (skipped by the layer's mask>0 check).
        while len(indices) < k:
            indices.append(0)
            weights.append(0.0)
        # Truncate if user gave more than k (keep largest weights).
        if len(indices) > k:
            order = sorted(range(len(indices)), key=lambda i: -weights[i])[:k]
            indices = [indices[i] for i in order]
            weights = [weights[i] for i in order]
        # Renormalize non-zero weights to sum to 1.
        s = sum(weights)
        if s > 0:
            weights = [w / s for w in weights]
        # Validate expert ids.
        for e in indices:
            if not (0 <= e < self.num_experts):
                raise ValueError(f"expert index {e} outside [0, {self.num_experts})")
        new_idx = torch.tensor([indices] * B, device=device, dtype=idx_dtype)
        new_wts = torch.tensor([weights] * B, device=device, dtype=wts_dtype)
        return new_idx, new_wts

    # -- obs frame dump ------------------------------------------------------

    def _dump_frames(self, batch: dict, chunk_idx: int) -> None:
        try:
            import numpy as np
            from PIL import Image
        except ImportError as e:
            print(f"[manual_routing] PIL/numpy not available: {e}")
            return

        self.image_dump_dir.mkdir(parents=True, exist_ok=True)
        for k, v in batch.items():
            kl = k.lower()
            if "image" not in kl and "camera" not in kl:
                continue
            arr = v
            if hasattr(arr, "detach"):
                arr = arr.detach().cpu().numpy()
            if arr.ndim == 4:  # (B, C, H, W) or (B, H, W, C)
                arr = arr[0]
            if arr.ndim != 3:
                continue
            if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                arr = arr.transpose(1, 2, 0)
            if arr.dtype.kind == "f":
                # Normalize whether it's [0, 1] or [-1, 1].
                lo, hi = float(arr.min()), float(arr.max())
                if lo >= -0.01 and hi <= 1.01:
                    arr = (arr * 255).clip(0, 255)
                else:
                    arr = ((arr - lo) / max(hi - lo, 1e-9) * 255).clip(0, 255)
                arr = arr.astype("uint8")
            elif arr.dtype != np.uint8:
                arr = arr.astype("uint8")
            stem = k.replace("/", "_").replace(".", "_")
            out = self.image_dump_dir / f"chunk_{chunk_idx:03d}_{stem}.png"
            Image.fromarray(arr).save(out)


# ---------------------------------------------------------------------------
# Wire into lerobot_eval
# ---------------------------------------------------------------------------

ROUTER: ManualRouter | None = None
_orig_run_one = lev.run_one


def _patched_run_one(task_group, task_id, env, **kw):
    global ROUTER
    pol = kw["policy"]
    if ROUTER is None:
        router_module = getattr(pol.model, "whole_expert_router", None)
        if router_module is None:
            raise RuntimeError(
                "policy.model.whole_expert_router is None — this script is for v5 "
                "(whole-expert) checkpoints only."
            )
        # Read top_k / num_experts off the router itself so we don't depend on
        # which config dataclass is in scope.
        top_k = int(router_module.top_k)
        num_experts = int(router_module.num_experts)
        image_dir = None
        if MANUAL_MODE == "interactive":
            image_dir = Path(os.environ.get("MANUAL_IMAGE_DIR", "/tmp/manual_routing"))
        ROUTER = ManualRouter(
            policy=pol,
            mode=MANUAL_MODE,
            program=PROGRAM,
            top_k=top_k,
            num_experts=num_experts,
            image_dump_dir=image_dir,
        )
        ROUTER.install()
        print(
            f"[manual_routing] installed mode={MANUAL_MODE} top_k={top_k} "
            f"num_experts={num_experts} program={MANUAL_PROGRAM_PATH or '-'}"
        )
    ROUTER.set_task(task_group, task_id)
    return _orig_run_one(task_group, task_id, env, **kw)


lev.run_one = _patched_run_one


def _dump():
    if ROUTER is None or not ROUTER.records or OUTPUT_DIR_PATH is None:
        print(
            f"[manual_routing] nothing to dump (router={ROUTER is not None}, "
            f"records={0 if ROUTER is None else len(ROUTER.records)}, "
            f"dir={OUTPUT_DIR_PATH})"
        )
        return
    OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
    log = OUTPUT_DIR_PATH / "routing_log.json"
    with open(log, "w") as f:
        json.dump(ROUTER.records, f)
    print(f"[manual_routing] wrote {len(ROUTER.records)} records to {log}")
    # Also dump the program (if any) and a tiny manifest for provenance.
    manifest = {
        "manual_mode": MANUAL_MODE,
        "manual_program_path": MANUAL_PROGRAM_PATH,
        "manual_random_seed": MANUAL_RANDOM_SEED,
        "n_records": len(ROUTER.records),
        "top_k": ROUTER.top_k,
        "num_experts": ROUTER.num_experts,
    }
    with open(OUTPUT_DIR_PATH / "manual_routing_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    if PROGRAM is not None:
        with open(OUTPUT_DIR_PATH / "manual_program_used.json", "w") as f:
            json.dump(PROGRAM, f, indent=2)


atexit.register(_dump)


if __name__ == "__main__":
    lev.eval_main()
