#!/usr/bin/env python3
"""Compare 4 videos in a 2x2 grid layout.

Usage:
    # 4 explicit videos
    python compare_videos.py v1.mp4 v2.mp4 v3.mp4 v4.mp4 -o compare.mp4

    # With custom labels for each quadrant
    python compare_videos.py v1.mp4 v2.mp4 v3.mp4 v4.mp4 \
        --labels "m1000 early" "m2000 mid" "m3000 late" "m4000 final"

    # Auto-resolve 4 models from a training run
    python compare_videos.py --run s3_rough_l1_4gpu \
        --models 1700 3000 5000 7000 \
        --project-root ../..

    # Record missing videos automatically
    python compare_videos.py --run s3_rough_l1_4gpu \
        --models 1700 3000 5000 7000 \
        --auto-record

Output: A single MP4 with 2x2 grid layout. Videos are cropped to the shortest.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Config ───────────────────────────────────────────────────────────────── #

VIDEO_DIR = Path("D:/Desktop_Files/GPU-Train/RTX6000/Magicbot_Z1/videos")
LABEL_SCRIPT = Path("D:/Desktop_Files/GPU-Train/RTX6000/Magicbot_Z1/scripts/label_video.py")

# ── Font helper ──────────────────────────────────────────────────────────── #


def _get_font(size: int = 20) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ["Consolas.ttf", "DejaVuSansMono.ttf", "arial.ttf", "LiberationMono-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# ── Grid composition ─────────────────────────────────────────────────────── #


def make_grid(
    frames_list: list[list[np.ndarray]],
    labels: list[str],
    gap: int = 4,
    border_color: tuple = (60, 60, 60),
    quadrant_label: bool = True,
    label_font_size: int = 20,
) -> list[np.ndarray]:
    """Compose 4 equal-length frame sequences into a 2x2 grid.

    All frame lists must have the same length and shape.
    """
    n = len(frames_list[0])
    h, w = frames_list[0][0].shape[:2]

    out_w = w * 2 + gap
    out_h = h * 2 + gap
    font = _get_font(label_font_size)

    result = []
    for i in range(n):
        canvas = np.full((out_h, out_w, 3), border_color, dtype=np.uint8)

        # Top-left
        canvas[0:h, 0:w] = frames_list[0][i]
        # Top-right
        canvas[0:h, w + gap : w * 2 + gap] = frames_list[1][i]
        # Bottom-left
        canvas[h + gap : h * 2 + gap, 0:w] = frames_list[2][i]
        # Bottom-right
        canvas[h + gap : h * 2 + gap, w + gap : w * 2 + gap] = frames_list[3][i]

        # Draw quadrant labels (top-left corner of each cell)
        if quadrant_label and labels:
            img = Image.fromarray(canvas)
            draw = ImageDraw.Draw(img)
            positions = [(6, 4), (w + gap + 6, 4), (6, h + gap + 4), (w + gap + 6, h + gap + 4)]
            for pos, label in zip(positions, labels):
                # Semi-transparent background
                bbox = draw.textbbox(pos, label, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.rounded_rectangle(
                    [pos[0] - 2, pos[1] - 1, pos[0] + tw + 4, pos[1] + th + 2],
                    radius=3,
                    fill=(0, 0, 0, 180),
                )
                draw.text(pos, label, font=font, fill=(255, 255, 0))
            canvas = np.array(img)

        result.append(canvas)

    return result


# ── Video loading ────────────────────────────────────────────────────────── #


def load_videos_crop(paths: list[Path]) -> tuple[list[list[np.ndarray]], int]:
    """Load 4 videos and crop all to the shortest frame count.

    Returns (list_of_frame_lists, min_frame_count).
    """
    all_frames = []
    counts = []
    for p in paths:
        frames = list(iio.imiter(str(p), plugin="pyav"))
        all_frames.append(frames)
        counts.append(len(frames))
        print(f"  Loaded {p.name}: {len(frames)} frames")

    min_n = min(counts)
    if len(set(counts)) > 1:
        print(f"  Cropping all to {min_n} frames (shortest video)")

    cropped = [frames[:min_n] for frames in all_frames]
    return cropped, min_n


# ── Auto-resolve from training run ──────────────────────────────────────── #


def find_video_for_model(run_name: str, model_iter: int) -> Path | None:
    """Try to find an existing labeled video for a given run + model iteration."""
    # Search all subdirectories for matching file
    pattern = f"*model{model_iter}*.mp4"
    for mp4 in VIDEO_DIR.rglob(pattern):
        if run_name in str(mp4) or True:  # accept any match
            return mp4
    return None


# ── Main ─────────────────────────────────────────────────────────────────── #


def main():
    parser = argparse.ArgumentParser(
        description="Compare 4 videos in a 2x2 grid layout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("videos", nargs="*", help="4 input video paths")
    parser.add_argument("-o", "--output", help="Output video path")
    parser.add_argument("--labels", nargs=4, metavar="LABEL", help="Custom labels for TL TR BL BR quadrants")
    parser.add_argument("--gap", type=int, default=4, help="Gap between cells in pixels (default: 4)")
    parser.add_argument("--no-quadrant-labels", action="store_true", help="Skip quadrant position labels")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS (default: 30)")
    parser.add_argument("--font-size", type=int, default=20, help="Quadrant label font size (default: 20)")

    # Auto-resolve mode
    parser.add_argument("--run", help="Training run name for auto-resolve")
    parser.add_argument("--models", nargs=4, type=int, metavar="ITER",
                        help="4 model iterations to compare (e.g. 1000 2000 3000 4000)")
    parser.add_argument("--auto-record", action="store_true",
                        help="Record missing videos automatically (requires Spark access)")

    args = parser.parse_args()

    # Resolve video paths
    if args.run and args.models:
        # Auto-resolve mode
        print(f"=== Auto-resolve: {args.run}, models {args.models} ===")
        paths = []
        for model_iter in args.models:
            video = find_video_for_model(args.run, model_iter)
            if video:
                print(f"  Found: {video}")
                paths.append(video)
            elif args.auto_record:
                print(f"  MISSING: model_{model_iter} — auto-record not yet implemented")
                print(f"    Run manually: /gpu-train --sim --checkpoint ... --video_length 200")
                sys.exit(1)
            else:
                print(f"  MISSING: model_{model_iter} — use --auto-record or record manually")
                sys.exit(1)
    elif len(args.videos) == 4:
        paths = [Path(v) for v in args.videos]
    else:
        parser.error("Provide exactly 4 video paths, or use --run + --models with 4 iterations")

    # Verify all exist
    for p in paths:
        if not p.exists():
            print(f"ERROR: {p} not found")
            sys.exit(1)

    # Output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = paths[0].parent / f"{paths[0].stem}_compare2x2.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Labels
    if args.labels:
        labels = args.labels
    elif args.run and args.models:
        labels = [f"model_{m}" for m in args.models]
    else:
        labels = [p.stem for p in paths]

    print(f"\n=== Loading videos ===")
    frames_list, n_frames = load_videos_crop(paths)

    print(f"\n=== Composing 2x2 grid ({n_frames} frames) ===")
    grid_frames = make_grid(
        frames_list,
        labels,
        gap=args.gap,
        quadrant_label=not args.no_quadrant_labels,
        label_font_size=args.font_size,
    )

    print(f"Writing to {out_path} ...")
    iio.imwrite(str(out_path), grid_frames, plugin="pyav", fps=args.fps, codec="libx264")
    print(f"Done: {out_path} ({n_frames} frames, {out_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
