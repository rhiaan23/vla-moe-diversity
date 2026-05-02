"""Annotate videos from a manual-routing eval with per-frame expert routing.

Adapted from ``annotate_eval_videos.py`` for the schema produced by
``eval_v5_manual_routing.py``. The routing_log.json records have:
  - ``manual_indices``    : (B, k) per-router-call manual choice
  - ``manual_weights``    : (B, k) renormalised weights
  - ``router_indices`` / ``router_weights`` : what the learned router
    *would* have picked, recorded for diff display
  - ``episode_batch``     : 1-indexed batch counter (per ``policy.reset()``)
  - ``chunk_idx``         : zero-indexed chunk position within an episode

The annotated overlay shows: prompt, frame index, chunk index, active
manual top-k, and (in muted text) what the router would have done. This
makes it visually obvious when a manual program is overriding the router.

Usage:
  python scripts/annotate_manual_routing_videos.py <eval_dir> [task_id]

Examples:
  python scripts/annotate_manual_routing_videos.py \\
    outputs/evals/manual_routing/family1_essential/phase_t5_approach_then_transport
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CHUNK = 50  # Pi0 chunk_size: one routing call covers 50 env steps

_PAL = [
    (0.267, 0.005, 0.329), (0.282, 0.140, 0.458), (0.254, 0.265, 0.530),
    (0.207, 0.372, 0.553), (0.164, 0.471, 0.558), (0.128, 0.567, 0.551),
    (0.135, 0.659, 0.518), (0.267, 0.749, 0.441), (0.478, 0.821, 0.318),
    (0.741, 0.873, 0.150), (0.993, 0.906, 0.144), (0.741, 0.150, 0.873),
    (0.873, 0.150, 0.741), (0.873, 0.150, 0.150), (0.150, 0.873, 0.150),
    (0.150, 0.150, 0.873),
]


def _ex_color(e: int) -> tuple[int, int, int]:
    return tuple(int(c) for c in np.array(_PAL[e % len(_PAL)]) * 255)


def _try_libero_prompts(suite: str) -> dict[int, str]:
    try:
        from libero.libero.benchmark import get_benchmark
        bench = get_benchmark(suite)()
        return {i: bench.get_task(i).language for i in range(bench.n_tasks)}
    except Exception as e:
        print(f"[warn] could not load libero prompts for {suite}: {e}")
        return {}


def main(out_dir: Path, only_task: str | None) -> None:
    log_path = out_dir / "routing_log.json"
    info_path = out_dir / "eval_info.json"
    if not log_path.exists() or not info_path.exists():
        sys.exit(f"missing routing_log.json or eval_info.json in {out_dir}")

    records = json.load(open(log_path))
    info = json.load(open(info_path))
    succ = {t["task_id"]: t["metrics"]["successes"] for t in info["per_task"]}
    n_eps_per_task = max(len(v) for v in succ.values()) if succ else 0
    suite = records[0]["group"] if records else "libero_10"
    prompts = _try_libero_prompts(suite)

    # Group records by (task, episode_batch); within batch, all envs share the
    # same chunk_idx series.
    by_te: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in records:
        by_te[(r["task"], r["episode_batch"])].append(r)
    for v in by_te.values():
        v.sort(key=lambda r: r["chunk_idx"])

    # Auto-detect batch size from the recorded batch dimension of indices.
    bs = len(records[0]["manual_indices"]) if records else 1
    print(
        f"detected suite={suite} batch_size={bs} n_episodes/task={n_eps_per_task} "
        f"records={len(records)} task_episodes_grouped={len(by_te)}"
    )

    annot_dir = out_dir / "videos_annotated"
    annot_dir.mkdir(parents=True, exist_ok=True)

    def expert_for_frame(task: int, ep: int, frame: int):
        """Return (manual_e1, manual_e2, manual_w1, router_e1, router_e2)."""
        ep_batch = ep // bs + 1
        row = ep % bs
        recs = by_te[(task, ep_batch)]
        if not recs:
            return -1, -1, 0.0, -1, -1
        chunk = min(frame // CHUNK, len(recs) - 1)
        rec = recs[chunk]
        mi = rec["manual_indices"][row]
        mw = rec["manual_weights"][row]
        ri = rec["router_indices"][row]
        m1 = int(mi[0])
        m2 = int(mi[1]) if len(mi) >= 2 else -1
        r1 = int(ri[0])
        r2 = int(ri[1]) if len(ri) >= 2 else -1
        return m1, m2, float(mw[0]), r1, r2

    def annotate_video(task: int, ep: int) -> Path | None:
        src = out_dir / "videos" / f"{suite}_{task}" / f"eval_episode_{ep}.mp4"
        if not src.exists():
            return None
        vid = iio.imread(src)  # (T, H, W, 3) uint8
        T, H, W, _ = vid.shape
        prompt = prompts.get(task, f"task {task}")
        success = succ.get(task, [False] * n_eps_per_task)[ep] if ep < n_eps_per_task else False
        bar_h = 78
        out_h = H + bar_h
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono-Bold.ttf", 11)
            font_sm = ImageFont.truetype("/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf", 10)
        except Exception:
            font = font_sm = ImageFont.load_default()

        out = np.zeros((T, out_h, W, 3), dtype=np.uint8)
        out[:, bar_h:, :, :] = vid
        for t in range(T):
            m1, m2, mw1, r1, r2 = expert_for_frame(task, ep, t)
            canvas = Image.fromarray(out[t])
            draw = ImageDraw.Draw(canvas)
            draw.rectangle([(0, 0), (W, bar_h)], fill=(0, 0, 0))
            flag = "SUCCESS" if success else "FAIL"
            flag_color = (60, 220, 80) if success else (220, 80, 80)
            wrap = (prompt[:34] + "...") if len(prompt) > 37 else prompt
            draw.text((4, 2), f"T{task}: {wrap}", fill=(220, 220, 220), font=font_sm)
            draw.text((W - 56, 2), flag, fill=flag_color, font=font_sm)
            chunk = t // CHUNK
            draw.text((4, 18), f"f={t:>3}/{T - 1}  chunk={chunk}", fill=(180, 180, 220), font=font)
            # Manual (foreground)
            line = f"manual: e{m1:>2}"
            if m2 >= 0:
                line += f"  e{m2:>2}"
            line += f"  w={mw1*100:>4.1f}%"
            draw.text((4, 34), line, fill=_ex_color(m1), font=font)
            # Router (muted)
            rline = f"router: e{r1:>2}"
            if r2 >= 0:
                rline += f"  e{r2:>2}"
            override = " (OVERRIDDEN)" if (m1 != r1 or m2 != r2) else ""
            draw.text((4, 50), rline + override, fill=(140, 140, 140), font=font_sm)
            out[t] = np.array(canvas)
        dst = annot_dir / f"{suite}_t{task}_ep{ep:02d}_{'PASS' if success else 'fail'}.mp4"
        iio.imwrite(dst, out, fps=30, codec="libx264", macro_block_size=1, quality=7)
        return dst

    n = 0
    for (tid, _eb) in sorted(by_te):
        if only_task is not None and str(tid) != only_task:
            continue
        for ep in range(min(20, n_eps_per_task)):
            dst = annotate_video(tid, ep)
            if dst:
                n += 1
                print(f"  [{n:>3}] {dst.name}")
    print(f"done: {n} annotated videos in {annot_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: annotate_manual_routing_videos.py <eval_dir> [task_id]")
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None)
