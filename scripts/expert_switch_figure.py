"""Expert-segment figure driven by chunk_routing.csv.

For each provided video, look up its (task, episode) row sequence in
`chunk_routing.csv`, group consecutive chunks with the same top-1 expert into
segments, sample 3 frames per segment (start / middle / 3-quarters through),
and lay them out as one continuous row per video. Borders are colored by the
top-1 expert active in that segment; legend at the bottom lists every expert
that appears in any plotted segment.

Usage:
    python scripts/expert_switch_figure.py \
        --chunk-csv outputs/eval/.../chunk_routing.csv \
        --video 4:1:outputs/eval/.../libero_10_t4_ep01_fail.mp4 \
        --video 2:2:outputs/eval/.../libero_10_t2_ep02_PASS.mp4 \
        --video 5:1:outputs/eval/.../libero_10_t5_ep01_PASS.mp4 \
        --out outputs/eval/.../expert_switch_figure.png

Each `--video` argument is `task:episode:path` (episode is 0-indexed, matching
the chunk_routing.csv `episode` column).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# 16-color palette consistent with scripts/annotate_v5_videos.py.
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


def expert_color(e: int) -> tuple[float, float, float]:
    return _PAL[e % len(_PAL)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunk-csv", type=Path, required=True)
    p.add_argument("--video", action="append", required=True,
                   help="task:episode:path  (repeat for multiple). 0-indexed episode.")
    p.add_argument("--frames", action="append", default=[],
                   help="task:episode:f1,f2,...  Explicit frame indices for this video. "
                        "Videos without an explicit list use evenly-spaced frames whose count "
                        "matches the longest explicit list (or 12 if no list is given).")
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--crop-top", type=int, default=60,
                   help="pixels to crop from top of each frame (annotation header). Default 60.")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def parse_video_args(specs: list[str]) -> list[tuple[int, int, Path]]:
    out = []
    for spec in specs:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            raise SystemExit(f"--video must be 'task:episode:path', got: {spec!r}")
        out.append((int(parts[0]), int(parts[1]), Path(parts[2]).resolve()))
    return out


def parse_frames_args(specs: list[str]) -> dict[tuple[int, int], list[int]]:
    out: dict[tuple[int, int], list[int]] = {}
    for spec in specs:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            raise SystemExit(f"--frames must be 'task:episode:f1,f2,...', got: {spec!r}")
        tid, ep = int(parts[0]), int(parts[1])
        idxs = [int(x) for x in parts[2].split(",") if x.strip() != ""]
        out[(tid, ep)] = idxs
    return out


def load_chunk_seq(csv_path: Path, task: int, ep: int) -> list[int]:
    """Return list of top-1 expert ids ordered by chunk index for one (task, ep)."""
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if int(r["task"]) == task and int(r["episode"]) == ep:
                rows.append((int(r["chunk"]), int(r["top1_expert"])))
    rows.sort()
    return [e for _, e in rows]


def build_segments(seq: list[int], chunk_size: int, T: int
                   ) -> list[tuple[int, int, int]]:
    """Group consecutive chunks with the same top-1 expert into segments,
    clipped to actual video length T. Returns (expert, start_frame, end_frame)."""
    if not seq or T <= 0:
        return []
    n_active_chunks = (T + chunk_size - 1) // chunk_size
    seq = seq[:n_active_chunks]
    segments = []
    seg_start = 0
    for c in range(1, len(seq)):
        if seq[c] != seq[seg_start]:
            segments.append((seq[seg_start], seg_start * chunk_size, c * chunk_size - 1))
            seg_start = c
    segments.append((seq[seg_start], seg_start * chunk_size, len(seq) * chunk_size - 1))
    return [(e, s, min(en, T - 1)) for e, s, en in segments]


def collect_segment_frames(video: np.ndarray, segments: list[tuple[int, int, int]]
                           ) -> list[tuple[int, int, np.ndarray]]:
    """For each segment, return (start, middle, near-end) frame samples ordered
    globally by frame index. Drops duplicates if a segment is shorter than 4 frames."""
    T = video.shape[0]
    out = []
    seen = set()
    for expert, s, e in segments:
        L = e - s + 1
        if L <= 0:
            continue
        offsets = [0, L // 2, (3 * L) // 4]
        for off in offsets:
            f = s + off
            if not (0 <= f < T) or f in seen:
                continue
            seen.add(f)
            out.append((f, expert, video[f]))
    out.sort(key=lambda x: x[0])
    return out


def add_border(frame: np.ndarray, color: tuple[float, float, float], width: int = 18,
               black_w: int = 6, is_first: bool = False, is_last: bool = False) -> np.ndarray:
    """Wrap frame in a uniform colored border then a thin black outer border.
    Colored: same thickness `width` on all four sides.
    Black (outside the colored): top/bottom = `black_w` (full); left/right =
    `black_w` for the row's first/last frame, else `black_w // 2` so adjacent
    half-shared black verticals combine into a single full-width line."""
    h, w, _ = frame.shape
    # Step 1: uniform colored border.
    h_c, w_c = h + 2 * width, w + 2 * width
    with_color = np.empty((h_c, w_c, 3), dtype=np.uint8)
    rgb = tuple(int(c * 255) for c in color)
    with_color[..., 0] = rgb[0]
    with_color[..., 1] = rgb[1]
    with_color[..., 2] = rgb[2]
    with_color[width:width + h, width:width + w, :] = frame

    # Step 2: thin white border outside the colored.
    b_left = black_w if is_first else black_w // 2
    b_right = black_w if is_last else black_w // 2
    h_b, w_b = h_c + 2 * black_w, w_c + b_left + b_right
    out = np.full((h_b, w_b, 3), 255, dtype=np.uint8)  # 255 = white
    out[black_w:black_w + h_c, b_left:b_left + w_c, :] = with_color
    return out


def expert_for_frame(seq: list[int], frame: int, chunk_size: int) -> int | None:
    """Look up the top-1 expert active at a given frame index."""
    if not seq:
        return None
    c = min(frame // chunk_size, len(seq) - 1)
    return seq[c]


def evenly_spaced(T: int, n: int) -> list[int]:
    """Return n frame indices evenly spaced over [0, T-1]."""
    if T <= 0 or n <= 0:
        return []
    if n == 1:
        return [0]
    return [int(round(i * (T - 1) / (n - 1))) for i in range(n)]


def main():
    args = parse_args()
    videos = parse_video_args(args.video)
    explicit = parse_frames_args(args.frames)

    # Determine target frame count: max length of any explicit frame list (or 12 fallback).
    target_n = max((len(v) for v in explicit.values()), default=12)

    per_episode = []  # (task, ep, video_path, picks, T)  picks=[(frame, expert, frame_rgb)]
    used_experts: set[int] = set()
    for tid, ep, vpath in videos:
        if not vpath.exists():
            print(f"  ! missing {vpath}")
            per_episode.append((tid, ep, vpath, [], 0))
            continue
        video = iio.imread(vpath)  # (T, H, W, 3) uint8
        T = video.shape[0]
        seq = load_chunk_seq(args.chunk_csv, tid, ep)
        if not seq:
            print(f"  ! no chunk_routing rows for task={tid} episode={ep}")
            per_episode.append((tid, ep, vpath, [], T))
            continue

        if (tid, ep) in explicit:
            frame_indices = [f for f in explicit[(tid, ep)] if 0 <= f < T]
            if len(frame_indices) != len(explicit[(tid, ep)]):
                dropped = [f for f in explicit[(tid, ep)] if not (0 <= f < T)]
                print(f"  warn task {tid} ep {ep}: {len(dropped)} explicit frames "
                      f"out of range [0,{T}) dropped: {dropped}")
        else:
            frame_indices = evenly_spaced(T, target_n)

        picks = []
        for f in frame_indices:
            e = expert_for_frame(seq, f, args.chunk_size)
            if e is None:
                continue
            used_experts.add(e)
            img = video[f]
            if args.crop_top > 0 and img.shape[0] > args.crop_top:
                img = img[args.crop_top:, :, :]
            picks.append((f, e, img))

        per_episode.append((tid, ep, vpath, picks, T))
        print(f"  task {tid} ep {ep}: T={T}  picked {len(picks)} frames "
              f"({'explicit' if (tid, ep) in explicit else 'evenly-spaced'}): "
              f"{[f for f, _, _ in picks]}")

    n_rows = len(per_episode)
    max_cols = max((len(p) for _, _, _, p, _ in per_episode), default=0)
    if max_cols == 0:
        print("[fig] no frames to plot — abort")
        return

    # Build one composite image per row by hstack-ing bordered frames; pad shorter
    # rows on the right with white. Then vstack rows with a small white separator.
    border_w = 18
    row_imgs: list[np.ndarray] = []
    h_per_frame = None
    for tid, ep, vpath, picks, T in per_episode:
        if not picks:
            continue
        bordered = []
        for i, (_, e, frm) in enumerate(picks):
            bordered.append(add_border(
                frm, expert_color(e), width=border_w,
                is_first=(i == 0), is_last=(i == len(picks) - 1),
            ))
        h_per_frame = bordered[0].shape[0]
        row_imgs.append(np.hstack(bordered))

    # Pad shorter rows on the right with white so all rows have equal width.
    target_w = max(img.shape[1] for img in row_imgs)
    for i, img in enumerate(row_imgs):
        if img.shape[1] < target_w:
            pad = np.full((img.shape[0], target_w - img.shape[1], 3), 255, dtype=np.uint8)
            row_imgs[i] = np.hstack([img, pad])

    sep_h = 18  # white separator between rows
    composite_pieces = []
    for i, img in enumerate(row_imgs):
        if i > 0:
            composite_pieces.append(np.full((sep_h, img.shape[1], 3), 255, dtype=np.uint8))
        composite_pieces.append(img)
    composite = np.vstack(composite_pieces)
    comp_h, comp_w, _ = composite.shape

    # Sizing: keep enough width per frame to be readable, plus space for left labels.
    in_per_frame = 1.4
    fig_w = in_per_frame * max_cols + 3.0
    composite_aspect = (n_rows * h_per_frame + (n_rows - 1) * sep_h) / target_w
    fig_h = composite_aspect * (fig_w - 3.0) + 3.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.08, 0.14, 0.91, 0.76])
    ax.imshow(composite, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Row labels on the left.
    for r_i, (tid, ep, vpath, picks, T) in enumerate(per_episode):
        if not picks:
            continue
        y_center = r_i * (h_per_frame + sep_h) + h_per_frame / 2
        ax.text(-12, y_center, f"task {tid}", ha="right", va="center",
                fontsize=22, fontweight="bold", clip_on=False)

    sorted_experts = sorted(used_experts)
    expert_label = {e: chr(ord("A") + i) for i, e in enumerate(sorted_experts)}
    legend_handles = [Patch(facecolor=expert_color(e), edgecolor="black",
                            label=f"expert {expert_label[e]}")
                      for e in sorted_experts]
    n_legend_cols = min(8, max(1, len(legend_handles)))
    fig.legend(handles=legend_handles, loc="lower center", ncol=n_legend_cols,
               frameon=False, fontsize=20, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Pi0 v5 (whole-expert MoE) — frames colored by top-1 expert at that chunk",
                 fontsize=22)

    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"[fig] saved {args.out}")


if __name__ == "__main__":
    main()
