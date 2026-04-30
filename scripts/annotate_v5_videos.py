"""Annotate v5 LIBERO-10 eval videos with per-frame expert routing.

Reads the routing log from `eval_v5_routing.py`, joins each video frame to the
router decision in effect for that timestep (chunk_size=50), and writes new
mp4s with a header overlay showing the task prompt, frame index, success
flag, and the top-1/top-2 expert at that moment.

Also rewrites `video_expert_map.csv` to include the natural-language prompt.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from libero.libero.benchmark import get_benchmark

OUT = Path("/scratch/gpfs/FHEIDE/rj2807/outputs/evals/v5_libero_10")
SUITE = "libero_10"
BS = 5
N_EP = 20
CHUNK = 50  # Pi0 chunk_size: one routing call covers 50 env steps
N_BATCHES = N_EP // BS
ANNOT_DIR = OUT / "videos_annotated"
ANNOT_DIR.mkdir(parents=True, exist_ok=True)

bench = get_benchmark(SUITE)()
prompts = {i: bench.get_task(i).language for i in range(bench.n_tasks)}

records = json.load(open(OUT / "routing_log.json"))
info = json.load(open(OUT / "eval_info.json"))
succ = {t["task_id"]: t["metrics"]["successes"] for t in info["per_task"]}

by_task: dict[int, list] = {}
for r in records:
    by_task.setdefault(r["task"], []).append(r)


def expert_for_frame(task: int, ep: int, frame: int) -> tuple[int, int, float]:
    """Return (top1, top2, top1_weight) for given frame of (task, ep)."""
    recs = by_task[task]
    calls_per_batch = len(recs) // N_BATCHES
    b = ep // BS
    row = ep % BS
    chunk = min(frame // CHUNK, calls_per_batch - 1)
    call = recs[b * calls_per_batch + chunk]
    idx = call["indices"][row]
    wts = call["weights"][row]
    return int(idx[0]), int(idx[1]), float(wts[0])


def annotate_video(task: int, ep: int) -> Path | None:
    src = OUT / "videos" / f"{SUITE}_{task}" / f"eval_episode_{ep}.mp4"
    if not src.exists():
        return None
    vid = iio.imread(src)  # (T, H, W, 3) uint8
    T, H, W, _ = vid.shape
    prompt = prompts[task]
    success = succ[task][ep]
    bar_h = 60  # extra space at top for text
    out_h = H + bar_h
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono-Bold.ttf", 11)
        font_sm = ImageFont.truetype("/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf", 10)
    except Exception:
        font = font_sm = ImageFont.load_default()

    out = np.zeros((T, out_h, W, 3), dtype=np.uint8)
    out[:, bar_h:, :, :] = vid
    for t in range(T):
        e1, e2, w1 = expert_for_frame(task, ep, t)
        canvas = Image.fromarray(out[t])
        draw = ImageDraw.Draw(canvas)
        # header background
        draw.rectangle([(0, 0), (W, bar_h)], fill=(0, 0, 0))
        # line 1: task + success flag
        flag = "SUCCESS" if success else "FAIL"
        flag_color = (60, 220, 80) if success else (220, 80, 80)
        wrap = (prompt[:34] + "...") if len(prompt) > 37 else prompt
        draw.text((4, 2), f"T{task}: {wrap}", fill=(220, 220, 220), font=font_sm)
        draw.text((W - 56, 2), flag, fill=flag_color, font=font_sm)
        # line 2: frame + experts
        draw.text((4, 18), f"f={t:>3}/{T - 1}  chunk={t // CHUNK}", fill=(180, 180, 220), font=font)
        # expert bar — color-coded by expert id (consistent palette)
        ex_color = lambda e: tuple(int(c) for c in np.array(_PAL[e % len(_PAL)]) * 255)
        draw.text((4, 34), f"top1: e{e1:>2} ({w1 * 100:>4.1f}%)", fill=ex_color(e1), font=font)
        draw.text((130, 34), f"top2: e{e2:>2}", fill=ex_color(e2), font=font)
        out[t] = np.array(canvas)
    dst = ANNOT_DIR / f"{SUITE}_t{task}_ep{ep:02d}_{'PASS' if success else 'fail'}.mp4"
    iio.imwrite(dst, out, fps=30, codec="libx264", macro_block_size=1, quality=7)
    return dst


# 16-color palette (matches viridis-ish, no seaborn dep)
_PAL = [
    (0.267, 0.005, 0.329), (0.282, 0.140, 0.458), (0.254, 0.265, 0.530),
    (0.207, 0.372, 0.553), (0.164, 0.471, 0.558), (0.128, 0.567, 0.551),
    (0.135, 0.659, 0.518), (0.267, 0.749, 0.441), (0.478, 0.821, 0.318),
    (0.741, 0.873, 0.150), (0.993, 0.906, 0.144), (0.741, 0.150, 0.873),
    (0.873, 0.150, 0.741), (0.873, 0.150, 0.150), (0.150, 0.873, 0.150),
    (0.150, 0.150, 0.873),
]


def update_csv():
    rows = []
    for tid in sorted(by_task):
        recs = by_task[tid]
        calls_per_batch = len(recs) // N_BATCHES
        for ep in range(min(10, N_EP)):  # only first 10 per task get videos
            b = ep // BS
            row = ep % BS
            ep_top1 = Counter()
            ep_top2 = Counter()
            for r in recs[b * calls_per_batch : (b + 1) * calls_per_batch]:
                ep_top1[r["indices"][row][0]] += 1
                ep_top2[r["indices"][row][1]] += 1
            t1, _ = ep_top1.most_common(1)[0]
            t2, _ = ep_top2.most_common(1)[0]
            ann = ANNOT_DIR / f"{SUITE}_t{tid}_ep{ep:02d}_{'PASS' if succ[tid][ep] else 'fail'}.mp4"
            raw = OUT / "videos" / f"{SUITE}_{tid}" / f"eval_episode_{ep}.mp4"
            rows.append({
                "task": tid, "episode": ep, "prompt": prompts[tid],
                "success": int(succ[tid][ep]),
                "top1_expert": t1, "top2_expert": t2,
                "raw_video": str(raw), "annotated_video": str(ann),
            })
    with open(OUT / "video_expert_map.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"updated {OUT / 'video_expert_map.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    import sys

    update_csv()
    only = sys.argv[1] if len(sys.argv) > 1 else None  # e.g. "5" to do task 5 only

    n = 0
    for tid in sorted(by_task):
        if only is not None and str(tid) != only:
            continue
        for ep in range(min(10, N_EP)):
            dst = annotate_video(tid, ep)
            if dst:
                n += 1
                print(f"  [{n:>3}] {dst.name}")
    print(f"done: {n} annotated videos in {ANNOT_DIR}")
