"""Analyze MuJoCo CSV data for p3_coarse local play."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
import sys
import os

# Read CSV
csv_path = sys.argv[1] if len(sys.argv) > 1 else None
if not csv_path:
    # Find latest CSV
    log_dir = "D:/Desktop_Files/GPU-Train/RTX6000/Magicbot_Z1/logs/p/p3"
    csvs = sorted([f for f in os.listdir(log_dir) if f.endswith(".csv")])
    csv_path = os.path.join(log_dir, csvs[-1])

print(f"Reading: {csv_path}")

with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

N = len(rows)
steps = np.arange(N)
time = np.array([float(r["time"]) for r in rows])
x = np.array([float(r["x"]) for r in rows])
y = np.array([float(r["y"]) for r in rows])
z = np.array([float(r["z"]) for r in rows])
fall = np.array([int(r["fall"]) for r in rows])

# Joint positions (in MuJoCo order)
jpos = {}
for jn in ["hip_p_l", "hip_r_l", "hip_y_l", "knee_p_l", "ank_p_l", "ank_r_l",
           "hip_p_r", "hip_r_r", "hip_y_r", "knee_p_r", "ank_p_r", "ank_r_r"]:
    jpos[jn] = np.array([float(r[f"qpos_{jn}"]) for r in rows])

# Actions (in MuJoCo order)
act = {}
for jn in ["hip_p_l", "hip_r_l", "hip_y_l", "knee_p_l", "ank_p_l", "ank_r_l",
           "hip_p_r", "hip_r_r", "hip_y_r", "knee_p_r", "ank_p_r", "ank_r_r"]:
    act[jn] = np.array([float(r[f"action_{jn}"]) for r in rows])

# Torques
tau = {}
for jn in ["hip_p_l", "hip_r_l", "hip_y_l", "knee_p_l", "ank_p_l", "ank_r_l",
           "hip_p_r", "hip_r_r", "hip_y_r", "knee_p_r", "ank_p_r", "ank_r_r"]:
    tau[jn] = np.array([float(r[f"tau_{jn}"]) for r in rows])

# Velocities
jvel = {}
for jn in ["hip_p_l", "knee_p_l", "ank_p_l", "hip_p_r", "knee_p_r", "ank_p_r"]:
    jvel[jn] = np.array([float(r[f"qvel_{jn}"]) for r in rows])

# === Analysis ===
print(f"\n=== Analysis ({N} steps, {time[-1]:.1f}s) ===")

# Forward velocity
dt = np.diff(time)
dt[dt == 0] = 0.02
vx = np.diff(x) / dt
vx = np.concatenate([[0], vx])
print(f"X displacement: {x[0]:.4f} -> {x[-1]:.4f} (total: {x[-1]-x[0]:.4f} m)")
print(f"Mean forward vel: {np.mean(vx[10:]):.3f} m/s (cmd: 0.5)")
print(f"Z height: mean={np.mean(z):.4f}, std={np.std(z):.4f}")

# Action statistics
print(f"\n=== Action Statistics ===")
for jn in ["hip_p_l", "knee_p_l", "ank_p_l", "hip_p_r", "knee_p_r", "ank_p_r"]:
    a = act[jn]
    print(f"  {jn:12s}: mean={np.mean(a):+.3f}  std={np.std(a):.3f}  range=[{np.min(a):+.3f}, {np.max(a):+.3f}]")

# Torque saturation check
print(f"\n=== Torque Saturation ===")
effort_hip = 120
effort_ankle = 50
for jn, lim in [("hip_p_l", effort_hip), ("knee_p_l", effort_hip), ("ank_p_l", effort_ankle),
                ("hip_p_r", effort_hip), ("knee_p_r", effort_hip), ("ank_p_r", effort_ankle)]:
    sat_pct = np.sum(np.abs(tau[jn]) > lim * 0.95) / N * 100
    print(f"  {jn:12s}: mean={np.mean(tau[jn]):+7.2f}  |max|={np.max(np.abs(tau[jn])):7.2f}  limit={lim}  saturated={sat_pct:.1f}%")

# Joint position range
print(f"\n=== Joint Position Range ===")
default = {
    "hip_p_l": -0.35, "hip_r_l": 0.0, "hip_y_l": 0.0, "knee_p_l": 0.7, "ank_p_l": -0.35, "ank_r_l": 0.0,
    "hip_p_r": -0.35, "hip_r_r": 0.0, "hip_y_r": 0.0, "knee_p_r": 0.7, "ank_p_r": -0.35, "ank_r_r": 0.0,
}
for jn in ["hip_p_l", "knee_p_l", "ank_p_l", "hip_p_r", "knee_p_r", "ank_p_r"]:
    jp = jpos[jn]
    print(f"  {jn:12s}: default={default[jn]:+.2f}  mean={np.mean(jp):+.4f}  range=[{np.min(jp):+.4f}, {np.max(jp):+.4f}]  swing={np.max(jp)-np.min(jp):.4f}")

# === Plotting ===
fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
fig.suptitle("p3_coarse model_7900 — Local MuJoCo Play Analysis", fontsize=14, fontweight="bold")

# 1. Forward displacement & velocity
ax = axes[0]
ax.plot(time, x, label="x pos (m)", linewidth=1)
ax.plot(time, vx, label="x vel (m/s)", alpha=0.7, linewidth=0.8)
ax.axhline(0.5, color="r", linestyle="--", alpha=0.5, label="cmd vel=0.5")
ax.set_ylabel("m / (m/s)")
ax.legend(loc="upper right")
ax.set_title("Forward Displacement & Velocity")
ax.grid(True, alpha=0.3)

# 2. Joint positions (left leg)
ax = axes[1]
for jn, color, lbl in [("hip_p_l", "blue", "hip_pitch_L"),
                         ("knee_p_l", "red", "knee_L"),
                         ("ank_p_l", "green", "ankle_pitch_L")]:
    ax.plot(time, jpos[jn], color=color, label=lbl, linewidth=0.8)
    ax.axhline(default[jn], color=color, linestyle=":", alpha=0.3)
for jn, color, lbl in [("hip_p_r", "blue", ""),
                         ("knee_p_r", "red", ""),
                         ("ank_p_r", "green", "")]:
    ax.plot(time, jpos[jn], color=color, linewidth=0.6, alpha=0.4)
ax.set_ylabel("rad")
ax.legend(loc="upper right")
ax.set_title("Joint Positions (solid=L, faded=R)")
ax.grid(True, alpha=0.3)

# 3. Actions
ax = axes[2]
for jn, color, lbl in [("hip_p_l", "blue", "act_hip_pitch"),
                         ("knee_p_l", "red", "act_knee"),
                         ("ank_p_l", "green", "act_ankle_pitch")]:
    ax.plot(time, act[jn], color=color, label=lbl, linewidth=0.8)
ax.set_ylabel("action")
ax.legend(loc="upper right")
ax.set_title("Policy Actions (left leg)")
ax.grid(True, alpha=0.3)

# 4. Torques
ax = axes[3]
for jn, color, lbl, lim in [("hip_p_l", "blue", "tau_hip_pitch", 120),
                              ("knee_p_l", "red", "tau_knee", 120),
                              ("ank_p_l", "green", "tau_ankle_pitch", 50)]:
    ax.plot(time, tau[jn], color=color, label=f"{lbl} (lim={lim})", linewidth=0.8)
    ax.axhline(lim, color=color, linestyle="--", alpha=0.2)
    ax.axhline(-lim, color=color, linestyle="--", alpha=0.2)
ax.set_ylabel("Nm")
ax.set_xlabel("time (s)")
ax.legend(loc="upper right")
ax.set_title("Joint Torques (left leg)")
ax.grid(True, alpha=0.3)

plt.tight_layout()

out_dir = os.path.dirname(csv_path)
out_path = os.path.join(out_dir, "p3_coarse_analysis.png")
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved: {out_path}")

# === Phase plot: hip_pitch vs knee (gait pattern) ===
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Gait Phase Plot", fontsize=13, fontweight="bold")

ax = axes2[0]
ax.plot(jpos["hip_p_l"], jpos["knee_p_l"], alpha=0.3, linewidth=0.5)
ax.scatter(jpos["hip_p_l"][0], jpos["knee_p_l"][0], c="green", s=50, zorder=5, label="start")
ax.set_xlabel("hip_pitch_L (rad)")
ax.set_ylabel("knee_L (rad)")
ax.set_title("Left Leg: hip_pitch vs knee")
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes2[1]
# Left vs right hip_pitch (gait symmetry)
ax.plot(jpos["hip_p_l"], jpos["hip_p_r"], alpha=0.3, linewidth=0.5)
ax.scatter(jpos["hip_p_l"][0], jpos["hip_p_r"][0], c="green", s=50, zorder=5, label="start")
lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, "k--", alpha=0.2, label="diagonal")
ax.set_xlabel("hip_pitch_L (rad)")
ax.set_ylabel("hip_pitch_R (rad)")
ax.set_title("Left vs Right hip_pitch")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
out_path2 = os.path.join(out_dir, "p3_coarse_gait.png")
fig2.savefig(out_path2, dpi=150)
print(f"Gait plot saved: {out_path2}")

plt.close("all")
