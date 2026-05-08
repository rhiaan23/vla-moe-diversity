"""Record-or-replay variant of eval_v5_routing.

Two phases, switched via the REPLAY_MODE env var:

- REPLAY_MODE=record: run a rollout of LIBERO task 0 with the native prompt;
  capture, per chunk inference, (z, router_indices, router_weights). Dump to
  <output_dir>/../replay_records.pt at exit. (One file per task — there is
  exactly one task and one episode.)

- REPLAY_MODE=replay: load that file, then run another rollout of the same
  task (same seed → same env init). At each chunk inference, replace the
  freshly-computed z and router decision with stored values from a permuted
  index. Permutation is "second half first": chunks
  [SPLIT, SPLIT+1, ..., N-1, 0, 1, ..., SPLIT-1], with SPLIT controlled by
  the REPLAY_PERM_SPLIT env var (default 4).

The point: test whether replaying the bottleneck's per-chunk control vector
in a different temporal order can drive the policy to do the chunks in that
new order — i.e., probe whether the 5-D ``z`` + sample-level expert choice
genuinely encode reusable control signals.
"""

from __future__ import annotations

import atexit
import os
import sys
from pathlib import Path

import torch

import lerobot.scripts.lerobot_eval as lev


MODE = os.environ.get("REPLAY_MODE", "record").lower()
SPLIT = int(os.environ.get("REPLAY_PERM_SPLIT", "4"))


def _output_dir_from_argv() -> Path | None:
    for i, a in enumerate(sys.argv):
        if a.startswith("--output_dir="):
            return Path(a.split("=", 1)[1])
        if a == "--output_dir" and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1])
    return None


_OUT = _output_dir_from_argv()
# Records live one level above the per-phase output dir so both phases can share.
_RECORDS_PATH = (_OUT.parent / "replay_records.pt") if _OUT is not None else None


_records: dict[str, list[torch.Tensor]] = {"z": [], "idx": [], "wts": []}
_counter = [0]
_permutation: list[int] | None = None


_orig_run_one = lev.run_one


def _install_record_hooks(model) -> None:
    projector = model.prefix_bottleneck_projector
    router = model.whole_expert_router
    if projector is None or router is None:
        raise RuntimeError(
            "replay_z_eval needs both prefix_bottleneck_projector and whole_expert_router on the policy model"
        )

    orig_project_z = projector.project_z
    orig_router_forward = router.forward

    def project_z_record(pooled):
        z = orig_project_z(pooled)
        _records["z"].append(z.detach().cpu())
        return z

    def router_record(ctx):
        idx, wts, aux = orig_router_forward(ctx)
        _records["idx"].append(idx.detach().cpu())
        _records["wts"].append(wts.detach().cpu())
        return idx, wts, aux

    projector.project_z = project_z_record  # type: ignore[assignment]
    router.forward = router_record  # type: ignore[assignment]


def _install_replay_hooks(model) -> None:
    if _RECORDS_PATH is None or not _RECORDS_PATH.exists():
        raise RuntimeError(f"Replay mode needs records at {_RECORDS_PATH} (run record phase first)")
    loaded = torch.load(_RECORDS_PATH, map_location="cpu", weights_only=False)
    n = len(loaded["z"])
    split = max(0, min(SPLIT, n))
    global _permutation
    _permutation = list(range(split, n)) + list(range(0, split))
    print(f"[replay] loaded {n} chunks; permutation = {_permutation}", flush=True)

    projector = model.prefix_bottleneck_projector
    router = model.whole_expert_router
    if projector is None or router is None:
        raise RuntimeError("replay_z_eval needs the bottleneck/whole-expert pieces on the model")

    num_experts = model.config.moe_num_experts

    z_dtype = projector.down[0].weight.dtype  # projector is kept in fp32

    # Audit log: prove which permuted index each replay chunk pulls from.
    audit: list[dict] = []

    def project_z_replay(pooled):
        i = _permutation[_counter[0]]
        z_src = loaded["z"][i]
        z = z_src.to(pooled.device, dtype=z_dtype)
        audit.append(
            {
                "replay_chunk": _counter[0],
                "src_chunk": i,
                "z_norm": float(z_src.float().norm().item()),
                "z_first5": z_src.float().flatten()[:5].tolist(),
            }
        )
        return z

    def router_replay(ctx):
        i = _permutation[_counter[0]]
        idx = loaded["idx"][i].to(ctx.device)
        wts = loaded["wts"][i].to(ctx.device)
        audit[-1]["idx"] = loaded["idx"][i].tolist()
        audit[-1]["wts_top"] = loaded["wts"][i].float().flatten()[:2].tolist()
        _counter[0] += 1  # advance once both hooks have fired this chunk
        expert_dtype = next(model.paligemma_with_expert.gemma_expert.parameters()).dtype
        wts = wts.to(dtype=expert_dtype)
        aux = {
            "load_balance_loss": torch.zeros((), device=ctx.device, dtype=torch.float32),
            "tokens_per_expert": torch.zeros(num_experts, device=ctx.device, dtype=torch.float32),
        }
        return idx, wts, aux

    # Stash audit list onto the model so atexit can dump it.
    model._replay_audit = audit  # type: ignore[attr-defined]

    projector.project_z = project_z_replay  # type: ignore[assignment]
    router.forward = router_replay  # type: ignore[assignment]


_replay_model_ref = [None]


def _patched_run_one(task_group, task_id, env, **kw):
    pol = kw["policy"]
    model = pol.model
    if MODE == "record":
        _install_record_hooks(model)
    elif MODE == "replay":
        _install_replay_hooks(model)
        _replay_model_ref[0] = model
    else:
        raise ValueError(f"Unknown REPLAY_MODE {MODE!r}; use 'record' or 'replay'.")
    return _orig_run_one(task_group, task_id, env, **kw)


lev.run_one = _patched_run_one


def _dump_records() -> None:
    if MODE != "record":
        return
    if _RECORDS_PATH is None:
        print(f"[record] no --output_dir on cli; skipping save (records held: {len(_records['z'])})")
        return
    _RECORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_records, _RECORDS_PATH)
    print(f"[record] saved {len(_records['z'])} chunk records to {_RECORDS_PATH}")
    # Also emit a routing_log.json compatible with eval_v5_routing so the
    # annotation script can overlay top1/top2 expert per frame.
    if _OUT is not None:
        import json

        rl = []
        for idx, wts in zip(_records["idx"], _records["wts"]):
            rl.append(
                {
                    "group": "libero_10",
                    "task": 0,
                    "indices": idx.tolist(),
                    "weights": wts.float().tolist(),
                }
            )
        rl_path = _OUT / "routing_log.json"
        rl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rl_path, "w") as f:
            json.dump(rl, f)
        print(f"[record] wrote {len(rl)} routing entries to {rl_path}")


def _dump_replay_audit() -> None:
    if MODE != "replay" or _OUT is None:
        return
    model = _replay_model_ref[0]
    if model is None or not hasattr(model, "_replay_audit"):
        return
    import json

    audit = model._replay_audit
    _OUT.mkdir(parents=True, exist_ok=True)
    with open(_OUT / "replay_audit.json", "w") as f:
        json.dump(audit, f, indent=2)
    print(f"[replay] wrote {len(audit)} audit entries to {_OUT / 'replay_audit.json'}")


atexit.register(_dump_records)
atexit.register(_dump_replay_audit)


if __name__ == "__main__":
    lev.eval_main()
