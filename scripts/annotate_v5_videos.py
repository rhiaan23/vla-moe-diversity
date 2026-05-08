"""Annotate LIBERO eval videos with a header overlay (prompt / frame index / success).

Optionally overlays per-frame expert routing when `routing_log.json` is present next to
`eval_info.json` (from `eval_v5_routing` runs).

For plain `lerobot_eval` output you only need `eval_info.json` and `videos/*/eval_episode_*.mp4`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 16-color palette (matches viridis-ish, no seaborn dep)
_PAL = [
    (0.267, 0.005, 0.329),
    (0.282, 0.140, 0.458),
    (0.254, 0.265, 0.530),
    (0.207, 0.372, 0.553),
    (0.164, 0.471, 0.558),
    (0.128, 0.567, 0.551),
    (0.135, 0.659, 0.518),
    (0.267, 0.749, 0.441),
    (0.478, 0.821, 0.318),
    (0.741, 0.873, 0.150),
    (0.993, 0.906, 0.144),
    (0.741, 0.150, 0.873),
    (0.873, 0.150, 0.741),
    (0.873, 0.150, 0.150),
    (0.150, 0.873, 0.150),
    (0.150, 0.150, 0.873),
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Eval output folder containing eval_info.json and videos/",
    )
    p.add_argument(
        "--suite",
        type=str,
        default="libero_10",
        help="LIBERO suite name used in video filenames (default: libero_10)",
    )
    p.add_argument(
        "--routing-log",
        type=Path,
        default=None,
        help="Path to routing_log.json (default: <eval-dir>/routing_log.json if present)",
    )
    p.add_argument(
        "--instruction",
        type=str,
        default="",
        help="Overlay this prompt instead of the benchmark task.language for all videos",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Vec env batch size used during routing eval (for expert timestamps only)",
    )
    p.add_argument(
        "--n-episodes",
        type=int,
        default=20,
        help="Episode count assumed for routing bookkeeping (matches original script default)",
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=50,
        help="Pi0 chunk_size (# env steps between routing recomputations)",
    )
    p.add_argument(
        "--max-episodes",
        type=int,
        default=10,
        help="Max episodes per task to annotate (matches lerobot_eval default rendered video count)",
    )
    return p.parse_args(argv)


def load_benchmark_prompts(suite: str) -> dict[int, str]:
    from libero.libero.benchmark import get_benchmark

    bench = get_benchmark(suite)()
    return {i: bench.get_task(i).language for i in range(bench.n_tasks)}


def build_routing_index(
    records: list | None,
    *,
    bs: int,
    n_ep: int,
    chunk: int,
) -> dict:
    n_batches = max(1, n_ep // bs) if bs else 1
    if not records:
        return {"__by_task": {}, "__n_batches": n_batches, "__chunk": chunk, "__bs": bs}
    by_task: dict[int, list] = {}
    for r in records:
        by_task.setdefault(int(r["task"]), []).append(r)
    return {"__by_task": by_task, "__n_batches": n_batches, "__chunk": chunk, "__bs": bs}


def expert_for_frame(
    by_task_inner: dict[int, list],
    task_id: int,
    ep: int,
    frame: int,
    *,
    n_batches: int,
    chunk: int,
    bs: int,
) -> tuple[int | None, int | None, float | None]:
    """Return (top1, top2, top1_weight); Nones if routing not available."""
    recs = by_task_inner.get(task_id)
    if not recs:
        return None, None, None
    calls_per_batch = len(recs) // n_batches if n_batches else 1
    if calls_per_batch < 1:
        return None, None, None
    b = ep // bs
    row = ep % bs
    chunk_idx = min(frame // chunk, calls_per_batch - 1)
    idx_in_recs = b * calls_per_batch + chunk_idx
    if idx_in_recs >= len(recs):
        return None, None, None
    call = recs[idx_in_recs]
    idx = call["indices"][row]
    wts = call["weights"][row]
    return int(idx[0]), int(idx[1]), float(wts[0])


def annotate_video(
    *,
    out: Path,
    suite: str,
    task_id: int,
    ep: int,
    prompt: str,
    success: bool,
    routing_state: dict,
    chunk: int,
) -> Path | None:
    src = out / "videos" / f"{suite}_{task_id}" / f"eval_episode_{ep}.mp4"
    if not src.exists():
        return None
    vid = iio.imread(src)  # (T, H, W, 3) uint8
    T, H, W, _ = vid.shape
    if T == 0:
        return None
    bar_h = 60
    out_h = H + bar_h
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono-Bold.ttf", 11
        )
        font_sm = ImageFont.truetype("/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf", 10)
    except Exception:
        font = font_sm = ImageFont.load_default()

    by_inner = routing_state.get("__by_task") or {}
    n_batches = int(routing_state.get("__n_batches") or 1)
    bs = int(routing_state.get("__bs") or 1)
    use_routing = bool(by_inner)

    out_arr = np.zeros((T, out_h, W, 3), dtype=np.uint8)
    out_arr[:, bar_h:, :, :] = vid
    for t in range(T):
        e1, e2, w1 = expert_for_frame(
            by_inner, task_id, ep, t, n_batches=n_batches, chunk=chunk, bs=bs
        )
        canvas = Image.fromarray(out_arr[t])
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([(0, 0), (W, bar_h)], fill=(0, 0, 0))
        flag = "SUCCESS" if success else "FAIL"
        flag_color = (60, 220, 80) if success else (220, 80, 80)
        wrap = (prompt[:34] + "...") if len(prompt) > 37 else prompt
        draw.text((4, 2), f"T{task_id}: {wrap}", fill=(220, 220, 220), font=font_sm)
        draw.text((W - 56, 2), flag, fill=flag_color, font=font_sm)
        draw.text((4, 18), f"f={t:>3}/{T - 1}  chunk={t // chunk}", fill=(180, 180, 220), font=font)
        ex_color = lambda e: tuple(int(c) for c in np.array(_PAL[e % len(_PAL)]) * 255)
        if use_routing and e1 is not None and w1 is not None:
            draw.text((4, 34), f"top1: e{e1:>2} ({w1 * 100:>4.1f}%)", fill=ex_color(e1), font=font)
            if e2 is not None:
                draw.text((130, 34), f"top2: e{e2:>2}", fill=ex_color(e2), font=font)
        else:
            draw.text((4, 34), "routing: n/a", fill=(160, 160, 160), font=font)
        out_arr[t] = np.array(canvas)

    annot_dir = out / "videos_annotated"
    annot_dir.mkdir(parents=True, exist_ok=True)
    dst = annot_dir / f"{suite}_t{task_id}_ep{ep:02d}_{'PASS' if success else 'fail'}.mp4"
    iio.imwrite(dst, out_arr, fps=30, codec="libx264", macro_block_size=1, quality=7)
    return dst


def update_csv(
    out: Path,
    suite: str,
    eval_info: dict,
    prompts: dict[int, str],
    routing_state: dict,
    *,
    n_ep_cap: int,
    bs: int,
    n_ep: int,
    chunk: int,
) -> None:
    by_inner = routing_state.get("__by_task") or {}
    n_batches = int(routing_state.get("__n_batches") or 1)
    rows = []
    for t in eval_info.get("per_task", []):
        tid = int(t["task_id"])
        succ_list = t["metrics"]["successes"]
        n = min(len(succ_list), n_ep_cap)
        calls_per_batch = (
            len(by_inner.get(tid, [])) // n_batches if by_inner.get(tid) and n_batches else 0
        )
        for ep in range(n):
            if by_inner.get(tid) and calls_per_batch > 0:
                b = ep // bs
                row = ep % bs
                ep_top1: Counter = Counter()
                ep_top2: Counter = Counter()
                recs = by_inner[tid]
                for r in recs[b * calls_per_batch : (b + 1) * calls_per_batch]:
                    ep_top1[r["indices"][row][0]] += 1
                    ep_top2[r["indices"][row][1]] += 1
                t1, _ = ep_top1.most_common(1)[0] if ep_top1 else (-1, 0)
                t2, _ = ep_top2.most_common(1)[0] if ep_top2 else (-1, 0)
            else:
                t1, t2 = -1, -1
            success = bool(succ_list[ep])
            ann = out / "videos_annotated" / f"{suite}_t{tid}_ep{ep:02d}_{'PASS' if success else 'fail'}.mp4"
            raw = out / "videos" / f"{suite}_{tid}" / f"eval_episode_{ep}.mp4"
            rows.append(
                {
                    "task": tid,
                    "episode": ep,
                    "prompt": prompts.get(tid, ""),
                    "success": int(success),
                    "top1_expert": t1,
                    "top2_expert": t2,
                    "raw_video": str(raw.resolve()),
                    "annotated_video": str(ann.resolve()),
                }
            )
    if not rows:
        return
    csv_path = out / "video_expert_map.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"updated {csv_path} ({len(rows)} rows)")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    out = args.eval_dir.resolve()
    info_path = out / "eval_info.json"
    if not info_path.is_file():
        print(f"Missing {info_path}", file=sys.stderr)
        return 1

    info = json.load(open(info_path))
    routing_path = args.routing_log if args.routing_log is not None else out / "routing_log.json"
    records = None
    if routing_path.is_file():
        records = json.load(open(routing_path))
    elif args.routing_log is not None:
        print(f"Missing routing log {routing_path}", file=sys.stderr)
        return 1

    routing_state = build_routing_index(
        records,
        bs=args.batch_size,
        n_ep=args.n_episodes,
        chunk=args.chunk,
    )

    prompts = load_benchmark_prompts(args.suite)
    if args.instruction.strip():
        for k in list(prompts.keys()):
            prompts[k] = args.instruction.strip()

    n_ep_cap = max(1, args.max_episodes)
    update_csv(
        out,
        args.suite,
        info,
        prompts,
        routing_state,
        n_ep_cap=n_ep_cap,
        bs=args.batch_size,
        n_ep=args.n_episodes,
        chunk=args.chunk,
    )

    n = 0
    for t in info.get("per_task", []):
        tid = int(t["task_id"])
        succ_list = t["metrics"]["successes"]
        prompt = prompts.get(tid, prompts.get(0, ""))
        for ep in range(min(len(succ_list), n_ep_cap)):
            dst = annotate_video(
                out=out,
                suite=args.suite,
                task_id=tid,
                ep=ep,
                prompt=prompt,
                success=bool(succ_list[ep]),
                routing_state=routing_state,
                chunk=args.chunk,
            )
            if dst:
                n += 1
                print(f"  [{n:>3}] {dst.name}")
    print(f"done: {n} annotated videos in {out / 'videos_annotated'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
