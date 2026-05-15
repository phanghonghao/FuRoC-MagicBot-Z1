"""
Plot left-right symmetry analysis from MuJoCo CSV data.
Usage: python scripts/plot_symmetry.py <csv_path> [--output <output_dir>]
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_csv(path):
    df = pd.read_csv(path)
    return df


def plot_symmetry(df, output_dir, title_prefix=""):
    os.makedirs(output_dir, exist_ok=True)

    # Define paired joints: (left, right, label)
    paired_joints = [
        ('hip_p', 'hip_pitch'),
        ('hip_r', 'hip_roll'),
        ('hip_y', 'hip_yaw'),
        ('knee_p', 'knee_pitch'),
        ('ank_p', 'ankle_pitch'),
        ('ank_r', 'ankle_roll'),
    ]

    steps = df['step'].values

    # ---- Figure 1: Torque comparison (left vs right) ----
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'{title_prefix} Joint Torque: Left vs Right', fontsize=14, fontweight='bold')

    for idx, (joint_prefix, label) in enumerate(paired_joints):
        ax = axes[idx // 2, idx % 2]
        tau_l = df[f'tau_{joint_prefix}_l'].values
        tau_r = df[f'tau_{joint_prefix}_r'].values

        ax.plot(steps, tau_l, alpha=0.7, linewidth=0.5, color='#2196F3', label=f'L (mean={np.abs(tau_l).mean():.1f})')
        ax.plot(steps, tau_r, alpha=0.7, linewidth=0.5, color='#F44336', label=f'R (mean={np.abs(tau_r).mean():.1f})')
        ax.set_title(f'{label}', fontsize=11)
        ax.set_xlabel('Step')
        ax.set_ylabel('Torque (Nm)')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '1_torque_left_right.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir}/1_torque_left_right.png")

    # ---- Figure 2: Torque asymmetry (|L - R|) over time ----
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'{title_prefix} Torque Asymmetry |τ_L - τ_R|', fontsize=14, fontweight='bold')

    for idx, (joint_prefix, label) in enumerate(paired_joints):
        ax = axes[idx // 2, idx % 2]
        tau_l = df[f'tau_{joint_prefix}_l'].values
        tau_r = df[f'tau_{joint_prefix}_r'].values
        asym = np.abs(tau_l - tau_r)

        ax.plot(steps, asym, alpha=0.6, linewidth=0.5, color='#FF9800')
        # rolling mean
        window = min(50, len(asym) // 5) if len(asym) > 25 else 1
        if window > 1:
            rolling = pd.Series(asym).rolling(window, center=True).mean().values
            ax.plot(steps, rolling, linewidth=2, color='#E91E63', label=f'Rolling avg ({window})')
        ax.axhline(y=np.mean(asym), color='gray', linestyle='--', alpha=0.5, label=f'Mean={np.mean(asym):.1f}')
        ax.set_title(f'{label}', fontsize=11)
        ax.set_xlabel('Step')
        ax.set_ylabel('|τ_L - τ_R| (Nm)')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '2_torque_asymmetry.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir}/2_torque_asymmetry.png")

    # ---- Figure 3: Action asymmetry ----
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'{title_prefix} Action Asymmetry |a_L - a_R|', fontsize=14, fontweight='bold')

    for idx, (joint_prefix, label) in enumerate(paired_joints):
        ax = axes[idx // 2, idx % 2]
        a_l = df[f'action_{joint_prefix}_l'].values
        a_r = df[f'action_{joint_prefix}_r'].values
        asym = np.abs(a_l - a_r)

        ax.plot(steps, asym, alpha=0.6, linewidth=0.5, color='#4CAF50')
        window = min(50, len(asym) // 5) if len(asym) > 25 else 1
        if window > 1:
            rolling = pd.Series(asym).rolling(window, center=True).mean().values
            ax.plot(steps, rolling, linewidth=2, color='#2196F3', label=f'Rolling avg ({window})')
        ax.axhline(y=np.mean(asym), color='gray', linestyle='--', alpha=0.5, label=f'Mean={np.mean(asym):.2f}')
        ax.set_title(f'{label}', fontsize=11)
        ax.set_xlabel('Step')
        ax.set_ylabel('|a_L - a_R|')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '3_action_asymmetry.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir}/3_action_asymmetry.png")

    # ---- Figure 4: Summary bar chart ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{title_prefix} Symmetry Summary', fontsize=14, fontweight='bold')

    labels_short = ['hip_p', 'hip_r', 'hip_y', 'knee_p', 'ank_p', 'ank_r']
    tau_means_l = []
    tau_means_r = []
    action_means_l = []
    action_means_r = []

    for jp in labels_short:
        tau_means_l.append(np.abs(df[f'tau_{jp}_l'].values).mean())
        tau_means_r.append(np.abs(df[f'tau_{jp}_r'].values).mean())
        action_means_l.append(np.abs(df[f'action_{jp}_l'].values).mean())
        action_means_r.append(np.abs(df[f'action_{jp}_r'].values).mean())

    x = np.arange(len(labels_short))
    width = 0.35

    bars1 = ax1.bar(x - width/2, tau_means_l, width, label='Left', color='#2196F3', alpha=0.8)
    bars2 = ax1.bar(x + width/2, tau_means_r, width, label='Right', color='#F44336', alpha=0.8)
    ax1.set_title('Mean |Torque| (Nm)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_short, rotation=30)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax1.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=7)

    bars3 = ax2.bar(x - width/2, action_means_l, width, label='Left', color='#2196F3', alpha=0.8)
    bars4 = ax2.bar(x + width/2, action_means_r, width, label='Right', color='#F44336', alpha=0.8)
    ax2.set_title('Mean |Action|')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_short, rotation=30)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    for bar in bars3:
        h = bar.get_height()
        ax2.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=7)
    for bar in bars4:
        h = bar.get_height()
        ax2.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '4_symmetry_summary.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir}/4_symmetry_summary.png")

    # ---- Figure 5: Qpos (joint angle) left vs right ----
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'{title_prefix} Joint Angle (qpos): Left vs Right', fontsize=14, fontweight='bold')

    for idx, (joint_prefix, label) in enumerate(paired_joints):
        ax = axes[idx // 2, idx % 2]
        qp_l = df[f'qpos_{joint_prefix}_l'].values
        qp_r = df[f'qpos_{joint_prefix}_r'].values

        ax.plot(steps, qp_l, alpha=0.7, linewidth=0.5, color='#2196F3', label='Left')
        ax.plot(steps, qp_r, alpha=0.7, linewidth=0.5, color='#F44336', label='Right')
        ax.set_title(f'{label}', fontsize=11)
        ax.set_xlabel('Step')
        ax.set_ylabel('Angle (rad)')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '5_qpos_left_right.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir}/5_qpos_left_right.png")


def detect_gait_cycles(df, signal_col='qpos_knee_p_l', min_distance=15):
    """Detect gait cycles from a periodic joint signal (e.g., knee pitch)."""
    signal = df[signal_col].values
    peaks, _ = find_peaks(signal, distance=min_distance)

    if len(peaks) < 2:
        return None, None, None

    # Each gait cycle: from one peak to the next
    cycles = []
    for i in range(len(peaks) - 1):
        start, end = peaks[i], peaks[i + 1]
        cycles.append((start, end))

    period = np.mean(np.diff(peaks))
    return cycles, peaks, period


def extract_phase_aligned(signal, cycles, n_points=50):
    """Extract phase-aligned (0→1) signal averaged over multiple cycles."""
    aligned_cycles = []
    for start, end in cycles:
        cycle_data = signal[start:end]
        if len(cycle_data) < 2:
            continue
        phase = np.linspace(0, 1, len(cycle_data))
        interp_fn = interp1d(phase, cycle_data, kind='linear')
        aligned = interp_fn(np.linspace(0, 1, n_points))
        aligned_cycles.append(aligned)

    if not aligned_cycles:
        return None
    return np.mean(aligned_cycles, axis=0)


def plot_phase_aligned(df, output_dir, title_prefix=""):
    """Figure 6: Phase-aligned symmetry analysis (the CORRECT way)."""
    os.makedirs(output_dir, exist_ok=True)

    paired_joints = [
        ('hip_p', 'hip_pitch'),
        ('hip_r', 'hip_roll'),
        ('hip_y', 'hip_yaw'),
        ('knee_p', 'knee_pitch'),
        ('ank_p', 'ankle_pitch'),
        ('ank_r', 'ankle_roll'),
    ]

    # Detect gait cycles from left knee (most periodic signal)
    cycles_l, peaks_l, period = detect_gait_cycles(df, 'qpos_knee_p_l')
    if cycles_l is None:
        print("WARNING: Could not detect gait cycles. Skipping phase-aligned plots.")
        return

    # Detect right leg cycles independently
    cycles_r, peaks_r, _ = detect_gait_cycles(df, 'qpos_knee_p_r')

    print(f"  Detected {len(cycles_l)} left cycles, {len(cycles_r)} right cycles, period={period:.1f} steps")

    # Compute phase offset between L and R peaks
    if len(peaks_l) > 1 and len(peaks_r) > 1:
        offsets = []
        for pl in peaks_l[:10]:
            nearest_pr = peaks_r[np.argmin(np.abs(peaks_r - pl))]
            offsets.append(nearest_pr - pl)
        mean_offset = np.mean(offsets)
        print(f"  Mean L→R peak offset: {mean_offset:.1f} steps (half-cycle={period/2:.1f})")

    N_PHASE = 50
    phase_axis = np.linspace(0, 1, N_PHASE)

    # ---- Figure 6: Phase-aligned torque comparison ----
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'{title_prefix} Phase-Aligned Torque (TRUE symmetry)\n'
                 f'Left cycles aligned with Right cycles by gait phase',
                 fontsize=13, fontweight='bold')

    for idx, (jp, label) in enumerate(paired_joints):
        ax = axes[idx // 2, idx % 2]
        tau_l = df[f'tau_{jp}_l'].values
        tau_r = df[f'tau_{jp}_r'].values

        # Phase-align using each leg's own gait cycles
        aligned_l = extract_phase_aligned(tau_l, cycles_l, N_PHASE)
        aligned_r = extract_phase_aligned(tau_r, cycles_r, N_PHASE)

        if aligned_l is None or aligned_r is None:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'{label}', fontsize=11)
            continue

        ax.plot(phase_axis, aligned_l, linewidth=2, color='#2196F3', label='Left')
        ax.plot(phase_axis, aligned_r, linewidth=2, color='#F44336', label='Right')
        ax.fill_between(phase_axis, aligned_l, aligned_r, alpha=0.15, color='gray')

        # Compute phase-aligned asymmetry
        phase_asym = np.mean(np.abs(aligned_l - aligned_r))
        mean_torque = np.mean(np.abs(aligned_l))
        pct = phase_asym / mean_torque * 100 if mean_torque > 0 else 0

        ax.set_title(f'{label}  (phase-aligned |Δ|={phase_asym:.1f} Nm, {pct:.0f}%)',
                     fontsize=10)
        ax.set_xlabel('Gait Phase (0→1)')
        ax.set_ylabel('Torque (Nm)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add stance/swing annotation
        ax.axvspan(0, 0.5, alpha=0.05, color='green')
        ax.axvspan(0.5, 1.0, alpha=0.05, color='orange')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '6_phase_aligned_torque.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir}/6_phase_aligned_torque.png")

    # ---- Figure 7: Phase-aligned action comparison ----
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'{title_prefix} Phase-Aligned Action (TRUE symmetry)',
                 fontsize=13, fontweight='bold')

    for idx, (jp, label) in enumerate(paired_joints):
        ax = axes[idx // 2, idx % 2]
        a_l = df[f'action_{jp}_l'].values
        a_r = df[f'action_{jp}_r'].values

        aligned_l = extract_phase_aligned(a_l, cycles_l, N_PHASE)
        aligned_r = extract_phase_aligned(a_r, cycles_r, N_PHASE)

        if aligned_l is None or aligned_r is None:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'{label}', fontsize=11)
            continue

        ax.plot(phase_axis, aligned_l, linewidth=2, color='#2196F3', label='Left')
        ax.plot(phase_axis, aligned_r, linewidth=2, color='#F44336', label='Right')
        ax.fill_between(phase_axis, aligned_l, aligned_r, alpha=0.15, color='gray')

        phase_asym = np.mean(np.abs(aligned_l - aligned_r))
        mean_action = np.mean(np.abs(aligned_l))
        pct = phase_asym / mean_action * 100 if mean_action > 0 else 0

        ax.set_title(f'{label}  (phase-aligned |Δ|={phase_asym:.3f}, {pct:.0f}%)',
                     fontsize=10)
        ax.set_xlabel('Gait Phase (0→1)')
        ax.set_ylabel('Action')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax.axvspan(0, 0.5, alpha=0.05, color='green')
        ax.axvspan(0.5, 1.0, alpha=0.05, color='orange')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '7_phase_aligned_action.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir}/7_phase_aligned_action.png")

    # ---- Figure 8: Phase-aligned qpos comparison ----
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'{title_prefix} Phase-Aligned Joint Angle (TRUE symmetry)',
                 fontsize=13, fontweight='bold')

    for idx, (jp, label) in enumerate(paired_joints):
        ax = axes[idx // 2, idx % 2]
        qp_l = df[f'qpos_{jp}_l'].values
        qp_r = df[f'qpos_{jp}_r'].values

        aligned_l = extract_phase_aligned(qp_l, cycles_l, N_PHASE)
        aligned_r = extract_phase_aligned(qp_r, cycles_r, N_PHASE)

        if aligned_l is None or aligned_r is None:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'{label}', fontsize=11)
            continue

        ax.plot(phase_axis, aligned_l, linewidth=2, color='#2196F3', label='Left')
        ax.plot(phase_axis, aligned_r, linewidth=2, color='#F44336', label='Right')
        ax.fill_between(phase_axis, aligned_l, aligned_r, alpha=0.15, color='gray')

        phase_asym = np.mean(np.abs(aligned_l - aligned_r))
        mean_qpos = np.mean(np.abs(aligned_l))
        pct = phase_asym / mean_qpos * 100 if mean_qpos > 0 else 0

        ax.set_title(f'{label}  (phase-aligned |Δ|={phase_asym:.4f} rad, {pct:.0f}%)',
                     fontsize=10)
        ax.set_xlabel('Gait Phase (0→1)')
        ax.set_ylabel('Angle (rad)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax.axvspan(0, 0.5, alpha=0.05, color='green')
        ax.axvspan(0.5, 1.0, alpha=0.05, color='orange')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '8_phase_aligned_qpos.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir}/8_phase_aligned_qpos.png")

    # ---- Figure 9: Raw vs Phase-aligned asymmetry comparison ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'{title_prefix} Raw vs Phase-Aligned Asymmetry Comparison',
                 fontsize=13, fontweight='bold')

    labels_short = ['hip_p', 'hip_r', 'hip_y', 'knee_p', 'ank_p', 'ank_r']

    # Compute raw and phase-aligned for torque
    raw_tau_asym = []
    phase_tau_asym = []
    raw_action_asym = []
    phase_action_asym = []

    for jp in labels_short:
        # Raw
        tau_l = df[f'tau_{jp}_l'].values
        tau_r = df[f'tau_{jp}_r'].values
        a_l = df[f'action_{jp}_l'].values
        a_r = df[f'action_{jp}_r'].values
        raw_tau_asym.append(np.mean(np.abs(tau_l - tau_r)))
        raw_action_asym.append(np.mean(np.abs(a_l - a_r)))

        # Phase-aligned
        al_tau_l = extract_phase_aligned(tau_l, cycles_l, N_PHASE)
        al_tau_r = extract_phase_aligned(tau_r, cycles_r, N_PHASE)
        al_a_l = extract_phase_aligned(a_l, cycles_l, N_PHASE)
        al_a_r = extract_phase_aligned(a_r, cycles_r, N_PHASE)
        if al_tau_l is not None and al_tau_r is not None:
            phase_tau_asym.append(np.mean(np.abs(al_tau_l - al_tau_r)))
        else:
            phase_tau_asym.append(0)
        if al_a_l is not None and al_a_r is not None:
            phase_action_asym.append(np.mean(np.abs(al_a_l - al_a_r)))
        else:
            phase_action_asym.append(0)

    x = np.arange(len(labels_short))
    width = 0.35

    ax1.bar(x - width/2, raw_tau_asym, width, label='Raw |τL-τR|', color='#FF9800', alpha=0.8)
    ax1.bar(x + width/2, phase_tau_asym, width, label='Phase-aligned |τL-τR|', color='#4CAF50', alpha=0.8)
    ax1.set_title('Torque Asymmetry (Nm)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_short, rotation=30)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    for i, (r, p) in enumerate(zip(raw_tau_asym, phase_tau_asym)):
        ratio = p / r * 100 if r > 0 else 0
        ax1.annotate(f'{ratio:.0f}%', xy=(i + width/2, p),
                     xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8, color='green')

    ax2.bar(x - width/2, raw_action_asym, width, label='Raw |aL-aR|', color='#FF9800', alpha=0.8)
    ax2.bar(x + width/2, phase_action_asym, width, label='Phase-aligned |aL-aR|', color='#4CAF50', alpha=0.8)
    ax2.set_title('Action Asymmetry')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_short, rotation=30)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    for i, (r, p) in enumerate(zip(raw_action_asym, phase_action_asym)):
        ratio = p / r * 100 if r > 0 else 0
        ax2.annotate(f'{ratio:.0f}%', xy=(i + width/2, p),
                     xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8, color='green')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '9_raw_vs_phase_aligned.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir}/9_raw_vs_phase_aligned.png")


def main():
    parser = argparse.ArgumentParser(description='Plot symmetry analysis from MuJoCo CSV')
    parser.add_argument('csv_path', help='Path to CSV file')
    parser.add_argument('--output', default=None, help='Output directory (default: same as CSV)')
    args = parser.parse_args()

    csv_path = args.csv_path
    if args.output is None:
        args.output = os.path.dirname(csv_path)

    df = load_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Extract phase name from CSV filename
    phase_name = os.path.splitext(os.path.basename(csv_path))[0]
    title_prefix = f"[{phase_name}]"

    print("\n--- Raw symmetry plots (1-5) ---")
    plot_symmetry(df, args.output, title_prefix)

    print("\n--- Phase-aligned symmetry plots (6-9) ---")
    plot_phase_aligned(df, args.output, title_prefix)

    print(f"\nAll plots saved to: {args.output}")


if __name__ == '__main__':
    main()
