#!/usr/bin/env python3
"""Record MuJoCo video at a fixed velocity (no sweep).

This is the pre-sweep recording behavior restored as a convenience wrapper.
Default velocity: 0.3 m/s.  Duration: 1000 steps (~20s @50Hz).

Usage:
    # Default: 0.3 m/s, 1000 steps
    python sim2sim/mujoco_record_fixed.py \\
        --policy logs/.../exported/policy.pt \\
        --record /tmp/output.mp4

    # Custom velocity and steps:
    python sim2sim/mujoco_record_fixed.py \\
        --policy logs/.../exported/policy.pt \\
        --record /tmp/output.mp4 \\
        --vel_x 0.5 --num_steps 2000

    # With terrain:
    python sim2sim/mujoco_record_fixed.py \\
        --policy logs/.../exported/policy.pt \\
        --record /tmp/output.mp4 --phase p3
"""

import subprocess
import sys
from pathlib import Path

# Fixed defaults
DEFAULT_VEL_X = 0.3
DEFAULT_NUM_STEPS = 1000
MJCF_PATH = "~/magicbot-z1_description/mjcf/MAGICBOTZ1.xml"
SCRIPT = Path(__file__).parent / "mujoco_manual.py"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fixed-velocity MuJoCo recording")
    parser.add_argument("--policy", required=True, help="Path to exported policy.pt")
    parser.add_argument("--record", required=True, help="Output video path")
    parser.add_argument("--mjcf", default=MJCF_PATH, help="MJCF XML path")
    parser.add_argument("--vel_x", type=float, default=DEFAULT_VEL_X,
                        help=f"Forward velocity (default: {DEFAULT_VEL_X} m/s)")
    parser.add_argument("--num_steps", type=int, default=DEFAULT_NUM_STEPS,
                        help=f"Number of steps (default: {DEFAULT_NUM_STEPS})")
    parser.add_argument("--phase", default=None, help="Phase ID for terrain selection")
    parser.add_argument("--terrain", default=None, help="Terrain type override")
    parser.add_argument("--deploy_cfg", default=None, help="Path to deploy.yaml")
    args = parser.parse_args()

    cmd = [
        sys.executable, str(SCRIPT),
        "--mjcf", args.mjcf,
        "--policy", args.policy,
        "--record", args.record,
        "--vel_x", str(args.vel_x),
        "--num_steps", str(args.num_steps),
    ]
    if args.phase:
        cmd += ["--phase", args.phase]
    if args.terrain:
        cmd += ["--terrain", args.terrain]
    if args.deploy_cfg:
        cmd += ["--deploy_cfg", args.deploy_cfg]

    print(f"[fixed] vel_x={args.vel_x} m/s, steps={args.num_steps}")
    print(f"[fixed] cmd: {' '.join(cmd)}")
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
