#!/usr/bin/env python3
"""Add parameter labels to training videos (top-right corner overlay).

Labels show behavioral metrics that are cross-run comparable:
  - time_out: episode completion ratio (higher = better)
  - ep_len: mean episode length out of 1000 max steps
  - bad_ori: bad_orientation ratio (lower = better, robot falls less)
  - vel_err: velocity tracking error in m/s (lower = more accurate)

Usage:
    python label_video.py input.mp4 output.mp4 \
        --run s4_gentle \
        --model model_47862 \
        --time-out 0.927 \
        --ep-len 966 \
        --bad-ori 0.073 \
        --vel-err 0.41

    # Batch: label all videos in a directory
    python label_video.py ../videos/s3_rough_l1/*.mp4 --outdir ../videos/labeled/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ── Label drawing ────────────────────────────────────────────────────────── #


def _get_font(size: int = 18) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a nice font, fall back to default."""
    font_names = [
        "DejaVuSansMono.ttf",
        "Consolas.ttf",
        "arial.ttf",
        "LiberationMono-Regular.ttf",
    ]
    for name in font_names:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_label(
    frame: np.ndarray,
    lines: list[str],
    position: str = "top-right",
    font_size: int = 18,
    padding: int = 8,
    bg_color: tuple = (0, 0, 0, 160),
    text_color: tuple = (255, 255, 255, 255),
) -> np.ndarray:
    """Draw semi-transparent label box on a single frame.

    Args:
        frame: HxWx3 or HxWx4 uint8 array.
        lines: Text lines to render.
        position: "top-right" (default).
        font_size: Font size in pixels.
        padding: Inner padding of the label box.

    Returns:
        Labeled frame (same shape as input).
    """
    img = Image.fromarray(frame).convert("RGBA")
    font = _get_font(font_size)

    # Measure text
    dummy_draw = ImageDraw.Draw(img)
    line_heights = []
    max_width = 0
    for line in lines:
        bbox = dummy_draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        line_heights.append(h)
        max_width = max(max_width, w)

    total_height = sum(line_heights) + padding * 2 + (len(lines) - 1) * 2
    box_w = max_width + padding * 2

    # Position
    img_w, img_h = img.size
    if position == "top-right":
        x0 = img_w - box_w - 10
        y0 = 10
    elif position == "top-left":
        x0 = 10
        y0 = 10
    else:
        x0 = img_w - box_w - 10
        y0 = 10

    # Draw background
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    ov_draw.rounded_rectangle(
        [x0, y0, x0 + box_w, y0 + total_height],
        radius=6,
        fill=bg_color,
    )
    img = Image.alpha_composite(img, overlay)

    # Draw text
    draw = ImageDraw.Draw(img)
    cy = y0 + padding
    for i, line in enumerate(lines):
        draw.text((x0 + padding, cy), line, font=font, fill=text_color)
        cy += line_heights[i] + 2

    # Convert back to RGB if original was 3-channel
    out = np.array(img)
    if frame.ndim == 3 and frame.shape[2] == 3:
        out = out[:, :, :3]
    return out


# ── Main pipeline ────────────────────────────────────────────────────────── #


def label_video(
    input_path: str,
    output_path: str,
    lines: list[str],
    font_size: int = 18,
) -> None:
    """Read video, add labels to every frame, write output."""
    input_path = str(input_path)
    output_path = str(output_path)

    print(f"[label_video] Input:  {input_path}")
    print(f"[label_video] Output: {output_path}")
    print(f"[label_video] Labels: {lines}")

    # Read all frames
    frames = list(iio.imiter(input_path, plugin="pyav"))
    n = len(frames)
    print(f"[label_video] Frames: {n}")

    # Process each frame
    labeled = []
    for i, frame in enumerate(frames):
        labeled.append(draw_label(frame, lines, font_size=font_size))
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  Labeled {i + 1}/{n} frames", end="\r")

    print()

    # Write output — preserve input fps for correct playback speed
    import imageio.v2 as iio2
    try:
        meta = iio2.immeta(input_path, plugin="pyav")
        fps = int(meta.get("fps", 50))
    except Exception:
        fps = 50  # fallback: match control frequency (50Hz)
    iio.imwrite(output_path, labeled, plugin="pyav", fps=fps, codec="libx264")
    print(f"[label_video] Done: {output_path}")


def build_lines(args: argparse.Namespace) -> list[str]:
    """Build label lines from CLI args.

    Labels are split into two sections:
      - Header: identification (Run, Model)
      - Behavioral metrics: cross-run comparable (time_out, ep_len, bad_ori, vel_err)
    """
    lines = []

    # ── Identification (header) ──────────────────────────────────────────
    run = args.run
    if run:
        lines.append(f"Run: {run}")

    model = args.model
    if model and not model.startswith("model_"):
        model = f"model_{model}"
    if model:
        lines.append(f"Model: {model}")

    # ── Behavioral metrics (cross-run comparable) ────────────────────────
    if args.time_out is not None:
        lines.append(f"time_out: {args.time_out:.1%}")
    if args.ep_len is not None:
        lines.append(f"ep_len: {args.ep_len:.0f}/1000")
    if args.bad_ori is not None:
        lines.append(f"bad_ori: {args.bad_ori:.1%}")
    if args.vel_err is not None:
        lines.append(f"vel_err: {args.vel_err:.2f} m/s")

    # Custom lines
    if args.extra:
        lines.extend(args.extra)

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Add parameter labels to training videos"
    )
    parser.add_argument("input", nargs="+", help="Input video path(s)")
    parser.add_argument("-o", "--output", help="Output video path (single file mode)")
    parser.add_argument("--outdir", help="Output directory (batch mode)")
    parser.add_argument("--model", help="Model name (e.g. model_5000)")
    parser.add_argument("--run", help="Run/version name")
    parser.add_argument("--time-out", type=float, help="time_out ratio (0-1)")
    parser.add_argument("--ep-len", type=float, help="Mean episode length")
    parser.add_argument("--bad-ori", type=float, help="bad_orientation ratio (0-1)")
    parser.add_argument("--vel-err", type=float, help="Velocity tracking error (m/s)")
    parser.add_argument("--extra", nargs="*", help="Additional custom label lines")
    parser.add_argument("--font-size", type=int, default=18, help="Font size (default: 18)")

    args = parser.parse_args()

    lines = build_lines(args)
    if not lines:
        print("No label info provided. Use --model, --run, --reward, etc.")
        sys.exit(1)

    inputs = [Path(p) for p in args.input]

    if len(inputs) == 1 and args.output:
        # Single file mode
        label_video(str(inputs[0]), args.output, lines, font_size=args.font_size)
    elif args.outdir:
        # Batch mode
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        for inp in inputs:
            out = outdir / f"{inp.stem}_labeled{inp.suffix}"
            label_video(str(inp), str(out), lines, font_size=args.font_size)
    else:
        # Default: _labeled suffix
        for inp in inputs:
            out = inp.parent / f"{inp.stem}_labeled{inp.suffix}"
            label_video(str(inp), str(out), lines, font_size=args.font_size)


if __name__ == "__main__":
    main()
