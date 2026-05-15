"""Plot left-right joint symmetry from MuJoCo CSV data.

Auto-detects L/R joint pairs from CSV headers. Supports single or multiple
CSV files for cross-run comparison.

Usage:
  # Single run
  python scripts/plot_asymmetry.py logs/p/p2_fine/20260515/p2_fine.csv

  # Compare two runs
  python scripts/plot_asymmetry.py logs/p/p2_coarse/.../p2_coarse.csv logs/p/p2_fine/.../p2_fine.csv

  # With explicit labels
  python scripts/plot_asymmetry.py csv1 csv2 --labels p2_coarse p2_fine

  # Select metrics
  python scripts/plot_asymmetry.py csv1 csv2 --metrics tau action

  # Specify output path
  python scripts/plot_asymmetry.py csv1 csv2 -o comparison.png

Metrics (detected from CSV headers):
  tau    — joint torques (Nm)
  action — policy actions
  qpos   — joint positions (rad)
  qvel   — joint velocities (rad/s)
"""

import argparse
import csv
import os
import re
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── CSV reading ──────────────────────────────────────────────────────────

def read_csv(path):
    with open(path, "r") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


# ── Auto-detect L/R pairs ───────────────────────────────────────────────

def detect_pairs(headers, prefix):
    """Find all (left, right) column pairs for a given metric prefix.

    Looks for columns matching f"{prefix}_{name}_l" and f"{prefix}_{name}_r".
    Returns OrderedDict: joint_name -> (left_col, right_col)
    """
    pattern = re.compile(rf"^{prefix}_(.+)_(l|r)$")
    lefts, rights = {}, {}
    for h in headers:
        m = pattern.match(h)
        if m:
            name, side = m.group(1), m.group(2)
            if side == "l":
                lefts[name] = h
            else:
                rights[name] = h
    pairs = OrderedDict()
    for name in lefts:
        if name in rights:
            pairs[name] = (lefts[name], rights[name])
    return pairs


# ── Compute asymmetry ───────────────────────────────────────────────────

def compute_asymmetry(rows, pairs):
    """Compute mean |L|, mean |R|, L/R ratio for each joint pair."""
    results = OrderedDict()
    for name, (col_l, col_r) in pairs.items():
        ml = np.mean([abs(r[col_l]) for r in rows])
        mr = np.mean([abs(r[col_r]) for r in rows])
        ratio = ml / mr if mr > 1e-6 else 0.0
        results[name] = {"mean_L": ml, "mean_R": mr, "ratio": ratio}
    return results


# ── Pretty joint names ──────────────────────────────────────────────────

JOINT_NAME_MAP = {
    "hip_p": "hip pitch",
    "hip_r": "hip roll",
    "hip_y": "hip yaw",
    "knee_p": "knee pitch",
    "ank_p": "ankle pitch",
    "ank_r": "ankle roll",
}

def pretty_name(raw):
    return JOINT_NAME_MAP.get(raw, raw.replace("_", " "))


# ── Plotting ────────────────────────────────────────────────────────────

def plot_comparison(all_data, labels, metrics, output_path):
    """Generate L/R ratio comparison plot.

    all_data: dict[label][metric] -> OrderedDict of joint asymmetry results
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]

    n_runs = len(labels)
    colors = ["#4CAF50", "#FF9800", "#2196F3", "#E91E63", "#9C27B0", "#00BCD4"]

    for ax, metric in zip(axes, metrics):
        # Collect joint names from first available run
        joint_names = []
        for lbl in labels:
            if metric in all_data[lbl]:
                joint_names = list(all_data[lbl][metric].keys())
                break

        if not joint_names:
            ax.set_title(f"{metric} — no data")
            continue

        n_joints = len(joint_names)
        x = np.arange(n_joints)
        width = 0.8 / max(n_runs, 1)
        offsets = np.arange(n_runs) - (n_runs - 1) / 2

        for i, lbl in enumerate(labels):
            if metric not in all_data[lbl]:
                continue
            ratios = [all_data[lbl][metric][j]["ratio"] for j in joint_names]
            ax.bar(x + offsets[i] * width, ratios, width * 0.9,
                   label=lbl, color=colors[i % len(colors)], alpha=0.85)

            # Annotate ratios
            for j_idx, r in enumerate(ratios):
                ax.annotate(f"{r:.2f}", (x[j_idx] + offsets[i] * width, r),
                            ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.axhline(1.0, color="k", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([pretty_name(j) for j in joint_names],
                           rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("L / R Ratio")
        ax.set_title(f"{metric.upper()} — L/R Ratio")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, max(2.0, ax.get_ylim()[1] * 1.1))

    fig.suptitle("Left-Right Joint Symmetry", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {output_path}")


def plot_single_run(data, label, metrics, output_path):
    """Generate detailed L vs R bar chart for a single run."""
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 2, figsize=(14, 4 * n_metrics))
    if n_metrics == 1:
        axes = axes.reshape(1, -1)

    for row, metric in enumerate(metrics):
        if metric not in data:
            continue
        joint_names = list(data[metric].keys())
        n = len(joint_names)
        x = np.arange(n)
        width = 0.35

        # Left: mean |L| vs mean |R|
        ax = axes[row][0]
        ml = [data[metric][j]["mean_L"] for j in joint_names]
        mr = [data[metric][j]["mean_R"] for j in joint_names]
        ax.bar(x - width / 2, ml, width, label="Left", color="steelblue")
        ax.bar(x + width / 2, mr, width, label="Right", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels([pretty_name(j) for j in joint_names],
                           rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(f"Mean |{metric}|")
        ax.set_title(f"{label} — {metric.upper()} L vs R")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Right: L/R ratio
        ax = axes[row][1]
        ratios = [data[metric][j]["ratio"] for j in joint_names]
        bar_colors = ["#4CAF50" if 0.9 <= r <= 1.1 else "#FF5722" for r in ratios]
        ax.bar(x, ratios, color=bar_colors, alpha=0.85)
        ax.axhline(1.0, color="k", linestyle="--", alpha=0.5)
        for i, r in enumerate(ratios):
            ax.annotate(f"{r:.2f}", (i, r), ha="center", va="bottom",
                        fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([pretty_name(j) for j in joint_names],
                           rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("L / R Ratio")
        ax.set_title(f"{label} — {metric.upper()} L/R Ratio (green=balanced)")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, max(2.0, max(ratios) * 1.2))

    fig.suptitle(f"Symmetry Analysis: {label}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {output_path}")


# ── Text summary ────────────────────────────────────────────────────────

def print_summary(all_data, labels, metrics):
    for metric in metrics:
        print(f"\n{'='*60}")
        print(f"  {metric.upper()} L/R Ratio")
        print(f"{'='*60}")
        header = f"{'Joint':<18s}"
        for lbl in labels:
            header += f" {lbl:>12s}"
        if len(labels) > 1:
            header += "  delta"
        print(header)
        print("-" * len(header))

        joint_names = list(all_data[labels[0]][metric].keys())
        for j in joint_names:
            line = f"{pretty_name(j):<18s}"
            ratios = []
            for lbl in labels:
                r = all_data[lbl][metric][j]["ratio"]
                ratios.append(r)
                line += f" {r:12.2f}"
            if len(ratios) > 1:
                line += f"  {ratios[-1]-ratios[0]:+6.2f}"
            print(line)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot left-right joint symmetry from MuJoCo CSV data")
    parser.add_argument("csv_files", nargs="+", help="CSV file path(s)")
    parser.add_argument("--labels", nargs="+", help="Run labels (default: basename)")
    parser.add_argument("--metrics", nargs="+",
                        default=["tau", "action"],
                        choices=["tau", "action", "qpos", "qvel"],
                        help="Metrics to analyze (default: tau action)")
    parser.add_argument("-o", "--output", help="Output PNG path (default: auto)")
    parser.add_argument("--mode", choices=["compare", "detail", "auto"],
                        default="auto",
                        help="compare=L/R ratio bars, detail=L vs R + ratio")
    args = parser.parse_args()

    # Validate inputs
    for p in args.csv_files:
        if not os.path.isfile(p):
            print(f"Error: file not found: {p}")
            sys.exit(1)

    # Default labels from filenames
    labels = args.labels or [os.path.splitext(os.path.basename(p))[0] for p in args.csv_files]

    if len(labels) != len(args.csv_files):
        print("Error: number of labels must match number of CSV files")
        sys.exit(1)

    # Read and compute
    all_data = {}
    headers = None
    for path, label in zip(args.csv_files, labels):
        rows = read_csv(path)
        if headers is None:
            headers = list(rows[0].keys())
        all_data[label] = {}
        for metric in args.metrics:
            pairs = detect_pairs(headers, metric)
            if pairs:
                all_data[label][metric] = compute_asymmetry(rows, pairs)
        print(f"  {label}: {len(rows)} steps")

    # Check we have data
    has_data = any(all_data[l] for l in labels)
    if not has_data:
        print("Error: no matching L/R pairs found in CSV headers")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        first_dir = os.path.dirname(args.csv_files[0])
        output_path = os.path.join(first_dir, "asymmetry.png")

    # Determine mode
    mode = args.mode
    if mode == "auto":
        mode = "compare" if len(args.csv_files) > 1 else "detail"

    # Plot
    if mode == "compare":
        plot_comparison(all_data, labels, args.metrics, output_path)
    else:
        for label, path in zip(labels, args.csv_files):
            out = output_path if len(labels) == 1 else os.path.join(
                os.path.dirname(path), f"asymmetry_{label}.png")
            plot_single_run(all_data[label], label, args.metrics, out)

    # Text summary
    print_summary(all_data, labels, args.metrics)


if __name__ == "__main__":
    main()
