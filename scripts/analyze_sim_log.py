#!/usr/bin/env python3
"""Analyze MuJoCo simulation CSV log and generate key plots.

Usage:
    python scripts/analyze_sim_log.py logs/p2_fine_test.csv
    python scripts/analyze_sim_log.py logs/p2_fine_test.csv --output_dir logs/plots
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

JOINT_NAMES = [
    "hip_p_l", "hip_r_l", "hip_y_l", "knee_p_l", "ank_p_l", "ank_r_l",
    "hip_p_r", "hip_r_r", "hip_y_r", "knee_p_r", "ank_p_r", "ank_r_r",
]

JOINT_GROUPS = {
    "Hip Pitch":  ("hip_p_l", "hip_p_r"),
    "Hip Roll":   ("hip_r_l", "hip_r_r"),
    "Hip Yaw":    ("hip_y_l", "hip_y_r"),
    "Knee Pitch": ("knee_p_l", "knee_p_r"),
    "Ank Pitch":  ("ank_p_l", "ank_p_r"),
    "Ank Roll":   ("ank_r_l", "ank_r_r"),
}

EFFORT_LIMITS = {
    "hip": 120, "knee": 120, "ank": 50,
}


def get_limit(name):
    for key, val in EFFORT_LIMITS.items():
        if key in name:
            return val
    return 120


def analyze(csv_path, output_dir=None):
    if output_dir is None:
        # Place plots alongside CSV in a plots/ subdirectory
        csv_dir = os.path.dirname(csv_path)
        csv_base = os.path.splitext(os.path.basename(csv_path))[0]
        output_dir = os.path.join(csv_dir, "plots") if csv_dir else "plots"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    n_rows = len(df)
    duration = df['time'].iloc[-1] if n_rows > 1 else 0
    n_falls = df['fall'].sum()

    # Use matplotlib with Agg backend (no GUI)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = 9
    plt.rcParams['figure.dpi'] = 150

    results = {
        "csv": csv_path,
        "rows": n_rows,
        "duration_s": round(duration, 2),
        "falls": int(n_falls),
        "distance_m": round(df['x'].iloc[-1], 3) if n_rows > 1 else 0,
        "plots": [],
        "warnings": [],
    }

    # --- Plot 1: Robot trajectory (x-y) ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df['x'], df['y'], linewidth=0.8)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'Trajectory ({duration:.1f}s, {n_falls} falls, {results["distance_m"]:.2f}m forward)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "01_trajectory.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    results["plots"].append(path)

    # --- Plot 2: Joint torques over time ---
    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
    for i, (group_name, (left, right)) in enumerate(JOINT_GROUPS.items()):
        ax = axes[i]
        tau_l = df[f'tau_{left}']
        tau_r = df[f'tau_{right}']
        ax.plot(df['time'], tau_l, label=f'{left}', alpha=0.8, linewidth=0.5)
        ax.plot(df['time'], tau_r, label=f'{right}', alpha=0.8, linewidth=0.5)
        limit = get_limit(left)
        ax.axhline(limit, color='red', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.axhline(-limit, color='red', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.set_ylabel('N·m')
        ax.set_title(f'{group_name} Torque (limit ±{limit})')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Joint Torques', fontsize=12, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(output_dir, "02_torques.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    results["plots"].append(path)

    # --- Plot 3: Torque distribution histogram ---
    fig, ax = plt.subplots(figsize=(8, 4))
    tau_cols = [f'tau_{n}' for n in JOINT_NAMES]
    tau_max = df[tau_cols].abs().max().max()
    bins = np.linspace(-130, 130, 60)
    for col in tau_cols:
        ax.hist(df[col], bins=bins, alpha=0.3, label=col, density=True)
    ax.axvline(120, color='red', linestyle='--', alpha=0.5, label='hip/knee limit ±120')
    ax.axvline(-120, color='red', linestyle='--', alpha=0.5)
    ax.axvline(50, color='orange', linestyle='--', alpha=0.5, label='ankle limit ±50')
    ax.axvline(-50, color='orange', linestyle='--', alpha=0.5)
    ax.set_xlabel('Torque (N·m)')
    ax.set_ylabel('Density')
    ax.set_title('Torque Distribution')
    ax.legend(fontsize=6, ncol=3)
    fig.tight_layout()
    path = os.path.join(output_dir, "03_torque_histogram.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    results["plots"].append(path)

    # --- Plot 4: Joint angles (gait pattern) ---
    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
    for i, (group_name, (left, right)) in enumerate(JOINT_GROUPS.items()):
        ax = axes[i]
        ax.plot(df['time'], df[f'qpos_{left}'], label=left, alpha=0.8, linewidth=0.5)
        ax.plot(df['time'], df[f'qpos_{right}'], label=right, alpha=0.8, linewidth=0.5)
        ax.set_ylabel('rad')
        ax.set_title(f'{group_name} Angle')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Joint Angles (Gait Pattern)', fontsize=12, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(output_dir, "04_joint_angles.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    results["plots"].append(path)

    # --- Plot 5: Policy actions ---
    fig, ax = plt.subplots(figsize=(12, 4))
    action_cols = [f'action_{n}' for n in JOINT_NAMES]
    for col in action_cols:
        ax.plot(df['time'], df[col], alpha=0.6, linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Action')
    ax.set_title(f'Policy Actions (mean_abs={df[action_cols].abs().mean().mean():.3f})')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path = os.path.join(output_dir, "05_actions.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    results["plots"].append(path)

    # --- Plot 6: Velocity commands + body height ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    ax1.plot(df['time'], df['cmd_vx'], label='cmd_vx', linewidth=1)
    ax1.plot(df['time'], df['cmd_vy'], label='cmd_vy', linewidth=1)
    ax1.plot(df['time'], df['cmd_vyaw'], label='cmd_vyaw', linewidth=1)
    ax1.set_ylabel('m/s, rad/s')
    ax1.set_title('Velocity Commands')
    ax1.legend(loc='upper right', fontsize=7)
    ax1.grid(True, alpha=0.2)
    ax2.plot(df['time'], df['z'], linewidth=0.8)
    ax2.axhline(0.69, color='green', linestyle='--', alpha=0.5, label='nominal (0.69m)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Height (m)')
    ax2.set_title('Body Height')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.2)
    fig.tight_layout()
    path = os.path.join(output_dir, "06_velocity_height.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    results["plots"].append(path)

    # --- Compute analysis summary ---
    tau_all = df[tau_cols]
    results["torque_mean"] = round(tau_all.abs().mean().mean(), 2)
    results["torque_max"] = round(tau_all.abs().max().max(), 2)
    results["action_mean_abs"] = round(df[action_cols].abs().mean().mean(), 4)

    # Saturation detection
    for col in tau_cols:
        limit = get_limit(col)
        saturation_pct = (df[col].abs() >= limit * 0.95).mean() * 100
        if saturation_pct > 5:
            results["warnings"].append(f"{col}: {saturation_pct:.1f}% time near limit (±{limit})")

    # Height stability
    z_std = df['z'].std()
    if z_std > 0.1:
        results["warnings"].append(f"Body height unstable: std={z_std:.3f}m")

    # Action smoothness (jitter = diff variance)
    action_diff = df[action_cols].diff().dropna()
    jitter = action_diff.std().mean()
    results["action_jitter"] = round(jitter, 4)
    if jitter > 0.3:
        results["warnings"].append(f"High action jitter: {jitter:.3f} (policy may be unstable)")

    # Print summary
    print(f"\n=== Simulation Analysis: {csv_path} ===")
    print(f"  Duration:    {duration:.1f}s ({n_rows} steps @ 50Hz)")
    print(f"  Distance:    {results['distance_m']:.2f}m forward")
    print(f"  Falls:       {n_falls}")
    print(f"  Mean torque: {results['torque_mean']:.1f} N·m")
    print(f"  Max torque:  {results['torque_max']:.1f} N·m")
    print(f"  Action mean: {results['action_mean_abs']:.3f}")
    print(f"  Action jitter: {results['action_jitter']:.3f}")
    if results["warnings"]:
        print(f"\n  [!] Warnings ({len(results['warnings'])}):")
        for w in results["warnings"]:
            print(f"    - {w}")
    else:
        print(f"\n  [OK] No warnings detected")
    print(f"\n  Plots saved to: {output_dir}/")
    for p in results["plots"]:
        print(f"    {os.path.basename(p)}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze MuJoCo simulation CSV log")
    parser.add_argument("csv", help="Path to CSV log file")
    parser.add_argument("--output_dir", default=None, help="Output directory for plots")
    args = parser.parse_args()
    analyze(args.csv, args.output_dir)
