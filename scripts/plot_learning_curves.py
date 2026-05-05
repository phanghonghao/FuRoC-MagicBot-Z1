#!/usr/bin/env python3
"""Plot Z1 training learning curves from TensorBoard event files.

Generates 4 plots:
  1. All runs reward comparison
  2. Reward decomposition (selected run)
  3. Termination reasons (selected run)
  4. Training efficiency (selected run)

Usage:
  python plot_learning_curves.py --log_root <path> --output_dir <path>
  python plot_learning_curves.py --log_root <path> --focus_run <dir_name>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Style ─────────────────────────────────────────────────────────────────── #

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
]

RUN_ALIASES = {
    "2026-04-30_04-53-17_s1_flat": "s1_flat",
    "2026-04-30_14-55-05_s1_stable": "s1_stable",
    "2026-05-01_01-21-35_s1_highspeed": "s1_highspeed",
    "2026-05-01_01-31-15_s3_rough_fail": "s3_rough_fail",
    "2026-05-01_04-44-07_s1_flat_retry": "s1_flat_retry",
    "2026-05-01_04-50-05_s2_gentle": "s2_gentle",
    "2026-05-01_07-04-35_s3_rough_l2": "s3_rough_l2",
    "2026-05-04_11-19-50_s3_rough_l1": "s3_rough_l1",
    "2026-05-04_12-30-56_s3_rough_l1_mgpu": "s3_rough_l1_mgpu",
    "2026-05-04_12-34-00_s3_rough_l1_mgpu_4gpu": "s3_rough_l1_mgpu_4gpu",
    "2026-05-04_12-40-26_s3_rough_l1_4gpu": "s3_rough_l1_4gpu",
    "2026-05-04_16-56-05_s4_full_terrain": "s4_full",
}

# Best model info (from best_models.json)
BEST_MODELS = {
    "s1_flat": {"iter": 3861, "reward": 47.33},
    "s1_stable": {"iter": 1555, "reward": 28.93},
    "s1_highspeed": {"iter": 2997, "reward": 30.11},
    "s3_rough_fail": {"iter": 1933, "reward": 1.85},
    "s1_flat_retry": {"iter": 3861, "reward": 47.33},
    "s2_gentle": {"iter": 47862, "reward": 47.06},
    "s3_rough_l2": {"iter": 32790, "reward": 38.04},
    "s3_rough_l1": {"iter": 1778, "reward": 5.86},
    "s3_rough_l1_4gpu": {"iter": 5032, "reward": 31.20},
    "s4_full": {"iter": None, "reward": None},  # still training
}

# Runs to skip (no data or test runs)
SKIP_RUNS = {
    "2026-05-04_11-58-36_s4_multigpu_test",
    "2026-05-04_11-58-40_s4_multigpu_test",
    "2026-05-04_11-58-41_s4_multigpu_test",
    "2026-05-04_12-04-00_s3_multigpu_test2gpu",
    "2026-05-04_12-08-02_s3_2gpu_test",
    "2026-05-05_02-03-13_s4_full_terrain",
    "2026-05-05_02-09-31_s4_full_terrain",
    "2026-05-05_02-32-45_s4_full_terrain",
}


# ── Data loading ──────────────────────────────────────────────────────────── #

def load_scalar(ea: EventAccumulator, tag: str) -> tuple[list[int], list[float]]:
    """Load a scalar tag as (steps, values)."""
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def load_run_data(run_path: str) -> dict:
    """Load all scalar data from a single run directory."""
    ea = EventAccumulator(run_path)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    data = {}
    for tag in tags:
        steps, values = load_scalar(ea, tag)
        data[tag] = {"steps": steps, "values": values}
    return data


def smooth(values: list[float], window: int = 50) -> list[float]:
    """Simple moving average smoothing."""
    if len(values) < window:
        return values
    arr = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="same")
    # Fix edges
    for i in range(window // 2):
        smoothed[i] = np.mean(arr[:i + window // 2 + 1])
        smoothed[-(i + 1)] = np.mean(arr[-(i + window // 2 + 1):])
    return smoothed.tolist()


# ── Plot 1: All runs reward comparison ────────────────────────────────────── #

def plot_reward_comparison(all_data: dict, output_dir: str):
    """Plot mean_reward for all runs on one figure."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Sort runs by peak reward (best first)
    run_order = []
    for alias, data in all_data.items():
        if "Train/mean_reward" in data:
            peak = max(data["Train/mean_reward"]["values"])
            run_order.append((peak, alias))
    run_order.sort(reverse=True)

    for i, (_, alias) in enumerate(run_order):
        data = all_data[alias]
        steps = data["Train/mean_reward"]["steps"]
        values = data["Train/mean_reward"]["values"]
        color = COLORS[i % len(COLORS)]

        # Smooth for cleaner line
        sv = smooth(values, window=max(1, len(values) // 200))

        ax.plot(steps, sv, color=color, linewidth=1.2, alpha=0.85, label=alias)

        # Annotate best model
        if alias in BEST_MODELS and BEST_MODELS[alias]["iter"] is not None:
            best_iter = BEST_MODELS[alias]["iter"]
            best_reward = BEST_MODELS[alias]["reward"]
            # Find closest step index
            idx = min(range(len(steps)), key=lambda j: abs(steps[j] - best_iter))
            ax.plot(steps[idx], sv[idx], "v", color=color, markersize=8, zorder=5)
            ax.annotate(
                f"best: {best_reward:.1f}",
                xy=(steps[idx], sv[idx]),
                xytext=(10, 5),
                textcoords="offset points",
                fontsize=7,
                color=color,
                fontweight="bold",
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Z1 12DOF — All Training Runs: Mean Reward Comparison")
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)
    ax.set_ylim(bottom=min(-10, ax.get_ylim()[0]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    fig.tight_layout()
    path = os.path.join(output_dir, "1_reward_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Plot 2: Reward decomposition ─────────────────────────────────────────── #

def plot_reward_decomposition(data: dict, alias: str, output_dir: str):
    """Plot reward components as stacked area chart + total."""
    reward_tags = [k for k in data if k.startswith("Episode_Reward/")]
    # Sort by absolute mean contribution
    tag_means = []
    for tag in reward_tags:
        vals = data[tag]["values"]
        tag_means.append((np.mean(np.abs(vals)), tag))
    tag_means.sort(reverse=True)

    # Pick top 10 reward components
    top_tags = [t for _, t in tag_means[:10]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])

    steps = data[top_tags[0]]["steps"]

    # Top: individual reward lines
    positive_tags = []
    negative_tags = []
    for tag in top_tags:
        name = tag.replace("Episode_Reward/", "")
        vals = smooth(data[tag]["values"], window=max(1, len(data[tag]["values"]) // 200))
        mean_v = np.mean(vals)
        if mean_v >= 0:
            positive_tags.append((name, vals, tag))
        else:
            negative_tags.append((name, vals, tag))

    for i, (name, vals, _) in enumerate(positive_tags):
        ax1.fill_between(steps, 0, vals, alpha=0.3, color=COLORS[i % len(COLORS)])
        ax1.plot(steps, vals, color=COLORS[i % len(COLORS)], linewidth=0.8, label=f"+{name}")

    offset = len(positive_tags)
    for i, (name, vals, _) in enumerate(negative_tags):
        ax1.fill_between(steps, 0, vals, alpha=0.2, color=COLORS[(offset + i) % len(COLORS)])
        ax1.plot(steps, vals, color=COLORS[(offset + i) % len(COLORS)], linewidth=0.8, label=name)

    # Total reward line
    total_reward = smooth(data["Train/mean_reward"]["values"], window=max(1, len(steps) // 200))
    ax1.plot(steps, total_reward, "k-", linewidth=2.0, label="Total reward", zorder=10)

    ax1.set_ylabel("Reward Value")
    ax1.set_title(f"{alias} — Reward Decomposition (Top 10 Components)")
    ax1.legend(loc="upper right", ncol=3, framealpha=0.9)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    # Bottom: curriculum
    if "Curriculum/terrain_levels" in data:
        terrain = smooth(data["Curriculum/terrain_levels"]["values"], window=max(1, len(steps) // 200))
        ax2.plot(steps, terrain, "brown", linewidth=1.5, label="Terrain level")
        ax2.set_ylabel("Terrain Level")
    if "Curriculum/lin_vel_cmd_levels" in data:
        vel = smooth(data["Curriculum/lin_vel_cmd_levels"]["values"], window=max(1, len(steps) // 200))
        ax2.plot(steps, vel, "teal", linewidth=1.5, label="Velocity cmd level")
        ax2.legend(loc="upper left")

    ax2.set_xlabel("Iteration")
    ax2.set_title("Curriculum Progress")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    # Annotate best model
    if alias in BEST_MODELS and BEST_MODELS[alias]["iter"] is not None:
        best_iter = BEST_MODELS[alias]["iter"]
        for a in [ax1, ax2]:
            a.axvline(x=best_iter, color="red", linestyle="--", alpha=0.6, linewidth=1.5)
        ax1.annotate(f"Best model\n(iter {best_iter})", xy=(best_iter, ax1.get_ylim()[1]),
                     fontsize=8, color="red", ha="center", va="top")

    fig.tight_layout()
    path = os.path.join(output_dir, f"2_reward_decomposition_{alias}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Plot 3: Termination reasons ───────────────────────────────────────────── #

def plot_termination(data: dict, alias: str, output_dir: str):
    """Plot termination reason ratios + episode length."""
    term_tags = [k for k in data if k.startswith("Episode_Termination/")]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[1, 1])

    steps = data[term_tags[0]]["steps"]
    colors_term = {"time_out": "#2ca02c", "bad_orientation": "#d62728",
                   "base_height": "#ff7f0e"}

    for tag in term_tags:
        name = tag.replace("Episode_Termination/", "")
        vals = smooth(data[tag]["values"], window=max(1, len(steps) // 200))
        color = colors_term.get(name, "#7f7f7f")
        ax1.plot(steps, vals, color=color, linewidth=1.2, label=name)

    ax1.set_ylabel("Termination Ratio")
    ax1.set_title(f"{alias} — Episode Termination Reasons")
    ax1.legend(loc="upper right")
    ax1.set_ylim(-0.05, 1.05)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    # Episode length
    if "Train/mean_episode_length" in data:
        eplen = smooth(data["Train/mean_episode_length"]["values"], window=max(1, len(steps) // 200))
        ax2.plot(steps, eplen, "steelblue", linewidth=1.5, label="Episode length")
        ax2.set_ylabel("Mean Episode Length (steps)")
        ax2.set_title("Episode Length Over Training")
        ax2.legend()

    ax2.set_xlabel("Iteration")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    # Best model annotation
    if alias in BEST_MODELS and BEST_MODELS[alias]["iter"] is not None:
        best_iter = BEST_MODELS[alias]["iter"]
        for a in [ax1, ax2]:
            a.axvline(x=best_iter, color="red", linestyle="--", alpha=0.6, linewidth=1.5)

    fig.tight_layout()
    path = os.path.join(output_dir, f"3_termination_{alias}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Plot 4: Training efficiency ───────────────────────────────────────────── #

def plot_efficiency(data: dict, alias: str, output_dir: str):
    """Plot throughput, collection/learning time, entropy, learning rate."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    steps = data.get("Perf/total_fps", {}).get("steps", [])

    # Top-left: Throughput
    ax = axes[0, 0]
    if "Perf/total_fps" in data:
        fps = data["Perf/total_fps"]["values"]
        ax.plot(steps[:len(fps)], fps, color="steelblue", linewidth=0.8, alpha=0.5)
        sfps = smooth(fps, window=max(1, len(fps) // 200))
        ax.plot(steps[:len(sfps)], sfps, color="navy", linewidth=1.5, label="Smoothed")
        ax.set_ylabel("Steps/sec")
        ax.set_title("Training Throughput (FPS)")
        ax.legend()

    # Top-right: Collection vs Learning time
    ax = axes[0, 1]
    if "Perf/collection time" in data and "Perf/learning_time" in data:
        ct = smooth(data["Perf/collection time"]["values"], window=max(1, len(steps) // 200))
        lt = smooth(data["Perf/learning_time"]["values"], window=max(1, len(steps) // 200))
        ct_steps = data["Perf/collection time"]["steps"]
        lt_steps = data["Perf/learning_time"]["steps"]
        ax.plot(ct_steps[:len(ct)], ct, color="#d62728", linewidth=1.2, label="Collection")
        ax.plot(lt_steps[:len(lt)], lt, color="#2ca02c", linewidth=1.2, label="Learning")
        ax.set_ylabel("Time (sec/iter)")
        ax.set_title("Collection vs Learning Time")
        ax.legend()

    # Bottom-left: Entropy & policy std
    ax = axes[1, 0]
    if "Loss/entropy" in data:
        ent = smooth(data["Loss/entropy"]["values"], window=max(1, len(steps) // 200))
        ent_steps = data["Loss/entropy"]["steps"]
        ax.plot(ent_steps[:len(ent)], ent, color="#9467bd", linewidth=1.5, label="Entropy")
        ax.set_ylabel("Entropy")
        ax.set_title("Policy Entropy (Exploration)")
        ax.legend()

    # Bottom-right: Learning rate
    ax = axes[1, 1]
    if "Loss/learning_rate" in data:
        lr = data["Loss/learning_rate"]["values"]
        lr_steps = data["Loss/learning_rate"]["steps"]
        ax.plot(lr_steps[:len(lr)], lr, color="#ff7f0e", linewidth=1.5)
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    for ax in axes.flat:
        ax.set_xlabel("Iteration")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    fig.suptitle(f"{alias} — Training Efficiency & Diagnostics", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, f"4_efficiency_{alias}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(description="Plot Z1 learning curves")
    parser.add_argument("--log_root", required=True, help="Path to log root dir")
    parser.add_argument("--output_dir", default=".", help="Output directory for PNGs")
    parser.add_argument("--focus_run", default=None,
                        help="Run dir name for detailed plots (plots 2-4). "
                             "Default: auto-select the run with most data points.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load all runs
    print("=== Loading TensorBoard data ===")
    all_data = {}
    for run_dir in sorted(os.listdir(args.log_root)):
        if run_dir in SKIP_RUNS:
            continue
        run_path = os.path.join(args.log_root, run_dir)
        if not os.path.isdir(run_path):
            continue
        events = [f for f in os.listdir(run_path) if f.startswith("events.out.tfevents")]
        if not events:
            continue

        alias = RUN_ALIASES.get(run_dir, run_dir[:30])
        print(f"  Loading {alias}...", end=" ", flush=True)
        data = load_run_data(run_path)
        n_pts = len(data.get("Train/mean_reward", {}).get("steps", []))
        print(f"{n_pts} pts")

        if n_pts > 100:  # skip tiny test runs
            all_data[alias] = data

    # Select focus run for detailed plots
    if args.focus_run:
        focus_alias = RUN_ALIASES.get(args.focus_run, args.focus_run)
    else:
        # Auto-select: run with most data points and highest peak reward
        best_alias = None
        best_score = -1
        for alias, data in all_data.items():
            if "Train/mean_reward" in data:
                peak = max(data["Train/mean_reward"]["values"])
                n_pts = len(data["Train/mean_reward"]["steps"])
                score = peak + n_pts * 0.001  # slight bonus for more data
                if score > best_score:
                    best_score = score
                    best_alias = alias
        focus_alias = best_alias

    print(f"\n  Focus run for detailed analysis: {focus_alias}")

    # Generate all 4 plots
    print("\n=== Generating plots ===")
    plot_reward_comparison(all_data, args.output_dir)

    if focus_alias and focus_alias in all_data:
        focus_data = all_data[focus_alias]
        plot_reward_decomposition(focus_data, focus_alias, args.output_dir)
        plot_termination(focus_data, focus_alias, args.output_dir)
        plot_efficiency(focus_data, focus_alias, args.output_dir)

    print(f"\n=== Done! All plots saved to {args.output_dir} ===")


if __name__ == "__main__":
    main()
