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
        position: "top-right" (default) or "top-left".
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


def draw_vel_badge(
    frame: np.ndarray,
    vel_text: str,
    font_size: int = 20,
) -> np.ndarray:
    """Draw velocity badge in the top-left corner.

    Visually distinct from the metrics label: blue accent bar on the left,
    slightly larger font, positioned at top-left with a small gap below
    any existing top-left content.

    Args:
        frame: HxWx3 or HxWx4 uint8 array.
        vel_text: Velocity display string (e.g. "vel: 0.0→1.0 m/s").
        font_size: Font size in pixels.

    Returns:
        Frame with velocity badge overlay.
    """
    img = Image.fromarray(frame).convert("RGBA")
    font = _get_font(font_size)

    # Measure text
    dummy_draw = ImageDraw.Draw(img)
    bbox = dummy_draw.textbbox((0, 0), vel_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    padding_x, padding_y = 10, 6
    bar_w = 4  # accent bar width
    box_w = text_w + padding_x * 2 + bar_w + 4
    box_h = text_h + padding_y * 2

    x0, y0 = 10, 10

    # Draw background
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    ov_draw.rounded_rectangle(
        [x0, y0, x0 + box_w, y0 + box_h],
        radius=6,
        fill=(0, 0, 0, 180),
    )
    # Blue accent bar on left edge
    ov_draw.rounded_rectangle(
        [x0, y0, x0 + bar_w, y0 + box_h],
        radius=2,
        fill=(0, 150, 255, 255),
    )
    img = Image.alpha_composite(img, overlay)

    # Draw text
    draw = ImageDraw.Draw(img)
    draw.text(
        (x0 + bar_w + 4 + padding_x - padding_x + 4, y0 + padding_y),
        vel_text,
        font=font,
        fill=(255, 255, 255, 255),
    )

    out = np.array(img)
    if frame.ndim == 3 and frame.shape[2] == 3:
        out = out[:, :, :3]
    return out


def _format_dynamic_vel_text(vx: float | None, vy: float | None, vyaw: float | None) -> str | None:
    """Format a compact velocity badge from per-frame commands."""
    parts = []
    if vx is not None:
        parts.append(f"vx={vx:.2f}")
    if vy is not None:
        parts.append(f"vy={vy:.2f}")
    if vyaw is not None:
        parts.append(f"wz={vyaw:.2f}")
    if not parts:
        return None
    return "cmd " + " ".join(parts)


def _build_dynamic_vel_text(frame_idx: int, sweep_data: dict) -> str | None:
    """Build per-frame velocity badge text from supported JSON formats."""
    if sweep_data.get("type") == "isaac_lab":
        def _pick(values: list[float | None] | None) -> float | None:
            if not values:
                return None
            idx = min(frame_idx, len(values) - 1)
            value = values[idx]
            return None if value is None else float(value)

        return _format_dynamic_vel_text(
            _pick(sweep_data.get("vel_x")),
            _pick(sweep_data.get("vel_y")),
            _pick(sweep_data.get("vel_yaw")),
        )

    steps_per_vel = sweep_data["steps_per_vel"]
    vel_list = [v["velocity"] for v in sweep_data["per_velocity"]]
    vel_idx = min(frame_idx // steps_per_vel, len(vel_list) - 1)
    return f"vel_x: {vel_list[vel_idx]:.1f} m/s"


# ── Main pipeline ────────────────────────────────────────────────────────── #


def label_video(
    input_path: str,
    output_path: str,
    lines: list[str],
    font_size: int = 18,
    vel_text: str | None = None,
    sweep_json: str | None = None,
) -> None:
    """Read video, add labels to every frame, write output.

    If sweep_json is provided, the velocity badge is drawn dynamically
    per frame based on the sweep's per-velocity step ranges.
    """
    input_path = str(input_path)
    output_path = str(output_path)

    # Parse dynamic velocity data for per-frame labeling
    sweep_data = None
    if sweep_json:
        with open(sweep_json, encoding="utf-8") as f:
            sweep_data = json.load(f)
        if sweep_data.get("type") == "isaac_lab":
            print(f"[label_video] Isaac velocity log: {sweep_data.get('num_frames', 0)} frames")
        else:
            steps_per_vel = sweep_data["steps_per_vel"]
            vel_list = [v["velocity"] for v in sweep_data["per_velocity"]]
            vel_min = sweep_data["vel_min"]
            vel_max = sweep_data["vel_max"]
            print(f"[label_video] Sweep: {len(vel_list)} velocities × {steps_per_vel} steps "
                  f"({vel_min:.1f} → {vel_max:.1f} m/s)")

    print(f"[label_video] Input:  {input_path}")
    print(f"[label_video] Output: {output_path}")
    print(f"[label_video] Labels: {lines}")
    if vel_text:
        print(f"[label_video] Vel badge (top-left): {vel_text}")

    # Read all frames
    frames = list(iio.imiter(input_path, plugin="pyav"))
    n = len(frames)
    print(f"[label_video] Frames: {n}")

    # Process each frame
    labeled = []
    for i, frame in enumerate(frames):
        out = draw_label(frame, lines, font_size=font_size)

        # Dynamic velocity from JSON, or static vel_text
        if sweep_data:
            cur_vel_text = _build_dynamic_vel_text(i, sweep_data)
            if cur_vel_text:
                out = draw_vel_badge(out, cur_vel_text)
        elif vel_text:
            out = draw_vel_badge(out, vel_text)

        labeled.append(out)
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

    # ── Velocity (drawn separately as top-left badge, not in main label) ─
    # vel is handled by draw_vel_badge(), not added to lines

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
    parser.add_argument("--vel", type=str, help="Velocity info (e.g. '0.5 m/s' or '0.0→1.0 m/s')")
    parser.add_argument("--sweep-json", type=str, help="Sweep metadata JSON for dynamic velocity labels")
    parser.add_argument("--extra", nargs="*", help="Additional custom label lines")
    parser.add_argument("--font-size", type=int, default=18, help="Font size (default: 18)")

    args = parser.parse_args()

    lines = build_lines(args)
    vel_text = None
    if hasattr(args, 'vel') and args.vel:
        vel_text = f"vel_x: {args.vel}"
    if not lines and not vel_text and not args.sweep_json:
        print("No label info provided. Use --model, --run, --vel, etc.")
        sys.exit(1)

    inputs = [Path(p) for p in args.input]

    if len(inputs) == 1 and args.output:
        # Single file mode
        label_video(str(inputs[0]), args.output, lines, font_size=args.font_size,
                    vel_text=vel_text, sweep_json=args.sweep_json)
    elif args.outdir:
        # Batch mode
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        for inp in inputs:
            out = outdir / f"{inp.stem}_labeled{inp.suffix}"
            label_video(str(inp), str(out), lines, font_size=args.font_size,
                        vel_text=vel_text, sweep_json=args.sweep_json)
    else:
        # Default: _labeled suffix
        for inp in inputs:
            out = inp.parent / f"{inp.stem}_labeled{inp.suffix}"
            label_video(str(inp), str(out), lines, font_size=args.font_size,
                        vel_text=vel_text, sweep_json=args.sweep_json)


if __name__ == "__main__":
    main()
