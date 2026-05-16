#!/usr/bin/env python3
"""
MuJoCo Sim-to-Sim Validation for MagicBot Z1 Locomotion Policy.

Loads the trained policy (JIT or ONNX) along with deploy.yaml config
and runs the policy in a MuJoCo simulation for validation before real robot deployment.

Usage:
    # Flat ground (default):
    python sim2sim/mujoco_manual.py \
        --mjcf ../magicbot-z1_description/mjcf/MAGICBOTZ1.xml \
        --policy logs/rsl_rl/.../exported/policy.pt \
        --deploy_cfg logs/rsl_rl/.../params/deploy.yaml

    # Phase-aware terrain (auto-selects terrain for p3b):
    python sim2sim/mujoco_manual.py \
        --mjcf ../magicbot-z1_description/mjcf/MAGICBOTZ1.xml \
        --policy policy.pt --deploy_cfg deploy.yaml \
        --phase p3b --keyboard

    # Explicit terrain override:
    python sim2sim/mujoco_manual.py \
        --mjcf ../magicbot-z1_description/mjcf/MAGICBOTZ1.xml \
        --policy policy.pt --deploy_cfg deploy.yaml \
        --terrain p3b --record /tmp/output.mp4

    # Manual joint control (no policy):
    python sim2sim/mujoco_manual.py \
        --mjcf ../magicbot-z1_description/mjcf/MAGICBOTZ1.xml \
        --manual --keyboard
"""

import argparse
import math
import os
import time

# Set EGL for offscreen rendering on Linux (Windows uses default wgl)
if os.name != "nt":
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import yaml

# Phase → terrain mapping for auto terrain selection
PHASE_TERRAIN = {
    "p1": None,       # flat ground
    "p2": None,       # flat ground
    "p3": "p3",       # gentle terrain
    "p3b": "p3b",     # intermediate terrain
    "p4": "p3b",      # rough → use p3b for sim2sim
}


# ---------------------------------------------------------------------------
# Terrain generation (matching Isaac Sim p3b config)
# ---------------------------------------------------------------------------
def generate_terrain_data(terrain_type="p3b", seed=42, difficulty=1.0):
    """Generate terrain heightmap matching Isaac Sim training config.

    Args:
        terrain_type: "p3" or "p3b"
        seed: Random seed for reproducibility
        difficulty: Scale factor for terrain height (0-1). At 1.0, full height.
                    In Isaac Lab, actual difficulty = terrain_level / (num_rows - 1).
                    For p3_coarse with level ~5/9, difficulty ≈ 0.625.

    Returns: (nrow, ncol, half_x, half_y, max_elev, hmap)
    """
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    H_SCALE = 0.1  # meters per grid cell (matches Isaac Sim horizontal_scale)

    if terrain_type == "p3":
        # p3: gentle terrain — flat 70% + random_grid 30% (height 0-0.4m)
        TERRAIN_L = 24.0
        TERRAIN_W = 8.0
        MAX_ELEV = 0.4  # match IsaacSim grid_height_range=(0.0, 0.4)
    elif terrain_type == "p3b":
        # p3b: intermediate terrain — flat 50% + random_grid 30% + stairs 10% + boxes 10%
        TERRAIN_L = 24.0
        TERRAIN_W = 8.0
        MAX_ELEV = 0.6
    else:
        raise ValueError(f"Unknown terrain type: {terrain_type}")

    ncol = int(TERRAIN_L / H_SCALE)
    nrow = int(TERRAIN_W / H_SCALE)
    rng = np.random.default_rng(seed)
    hmap = np.zeros((nrow, ncol), dtype=np.float64)

    def fill_random_grid(r0, r1, c0, c1, h_range=(0.0, 0.6)):
        gw = max(2, int(0.45 / H_SCALE))
        for i in range(r0, r1 - gw + 1, gw):
            for j in range(c0, c1 - gw + 1, gw):
                h = rng.uniform(h_range[0], h_range[1]) / MAX_ELEV
                hmap[i:i+gw, j:j+gw] = h

    def fill_stairs(r0, r1, c0, c1):
        sw = max(2, int(0.3 / H_SCALE))
        sh = rng.uniform(0.05, 0.23)
        n_steps = (c1 - c0) // sw
        for s in range(n_steps):
            h = min((s + 1) * sh / MAX_ELEV, 1.0)
            hmap[r0:r1, c0+s*sw:min(c0+(s+1)*sw, c1)] = h

    def fill_boxes(r0, r1, c0, c1):
        gw = max(2, int(0.45 / H_SCALE))
        for i in range(r0, r1 - gw + 1, gw):
            for j in range(c0, c1 - gw + 1, gw):
                h = rng.uniform(0.0, 0.4) / MAX_ELEV
                hmap[i:i+gw, j:j+gw] = h

    # Layout depends on terrain type
    if terrain_type == "p3":
        # p3: gentle — flat 70% + random_grid 30%, alternating sections
        sections = [
            (0,              ncol // 3,     "flat"),
            (ncol // 3,      ncol // 2,     "random_grid"),
            (ncol // 2,      5 * ncol // 6, "flat"),
            (5 * ncol // 6,  ncol,          "random_grid"),
        ]
    elif terrain_type == "p3b":
        # p3b: intermediate — full mix
        sections = [
            (0,              ncol // 6,      "flat"),
            (ncol // 6,      ncol // 3,      "random_grid"),
            (ncol // 3,      ncol // 2,      "stairs"),
            (ncol // 2,      2 * ncol // 3,  "flat"),
            (2 * ncol // 3,  5 * ncol // 6,  "boxes"),
            (5 * ncol // 6,  ncol,           "random_grid"),
        ]

    for c0, c1, stype in sections:
        if stype == "random_grid":
            if terrain_type == "p3":
                fill_random_grid(0, nrow, c0, c1, h_range=(0.0, 0.4))
            else:
                fill_random_grid(0, nrow, c0, c1)
        elif stype == "stairs":
            fill_stairs(0, nrow, c0, c1)
        elif stype == "boxes":
            fill_boxes(0, nrow, c0, c1)

    # Light smoothing (2 passes of 3x3 box filter)
    for _ in range(2):
        padded = np.pad(hmap, 1, mode='edge')
        hmap = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
            padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
            padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        ) / 9.0

    # Apply difficulty scaling (match Isaac Lab curriculum level)
    if difficulty < 1.0:
        hmap *= difficulty
        effective_max = MAX_ELEV * difficulty
        print(f"[TERRAIN] Difficulty={difficulty:.3f}: scaling height to {effective_max:.3f}m "
              f"(was {MAX_ELEV:.3f}m)")
    else:
        effective_max = MAX_ELEV

    print(f"[TERRAIN] Generated {terrain_type}: {TERRAIN_L}m x {TERRAIN_W}m, "
          f"grid {nrow}x{ncol}, elevation [{hmap.min()*MAX_ELEV:.3f}, {hmap.max()*MAX_ELEV:.3f}]m, "
          f"effective_max={effective_max:.3f}m")
    return nrow, ncol, TERRAIN_L / 2, TERRAIN_W / 2, effective_max, hmap


def load_model_with_terrain(mjcf_path, terrain_type, difficulty=1.0):
    """Load MuJoCo model with terrain hfield injected via Python API."""
    import mujoco

    nrow, ncol, half_x, half_y, max_elev, terrain_data = generate_terrain_data(terrain_type, difficulty=difficulty)

    # Read original XML
    with open(mjcf_path, 'r') as f:
        xml = f.read()

    # Make meshdir absolute (since we'll load from string, not file)
    mjcf_dir = os.path.dirname(os.path.abspath(mjcf_path))
    xml = xml.replace('meshdir="../meshes/"', f'meshdir="{mjcf_dir}/../meshes/"')

    # Replace ground plane with hfield geom
    old_ground = '<geom name="ground" type="plane" pos="0 0 0" friction="1 1 5" size="10 10 1" conaffinity="1" contype="1" material="MatGnd"/>'
    new_ground = '<geom name="ground" type="hfield" hfield="terrain_hf" pos="0 0 0" friction="1 1 5" conaffinity="1" contype="1"/>'
    xml = xml.replace(old_ground, new_ground)

    # Add hfield asset (no file - data set via Python API)
    hfield_line = f'        <hfield name="terrain_hf" nrow="{nrow}" ncol="{ncol}" size="{half_x} {half_y} {max_elev} 0.001"/>\n'
    xml = xml.replace('</asset>\n', hfield_line + '</asset>\n')

    model = mujoco.MjModel.from_xml_string(xml)
    model.hfield_data[:] = terrain_data.flatten()
    return model


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="MuJoCo sim-to-sim deployment for MagicBot Z1")
    parser.add_argument("--mjcf", type=str, required=True, help="Path to MAGICBOTZ1.xml")
    parser.add_argument("--policy", type=str, default=None, help="Path to policy file (.pt or .onnx)")
    parser.add_argument("--deploy_cfg", type=str, default=None, help="Path to deploy.yaml (optional)")
    parser.add_argument("--manual", action="store_true", help="Manual joint control mode (no policy)")
    parser.add_argument("--onnx", action="store_true", help="Use ONNX model instead of JIT")
    parser.add_argument("--vel_x", type=float, default=0.5, help="Forward velocity command (m/s)")
    parser.add_argument("--vel_y", type=float, default=0.0, help="Lateral velocity command (m/s)")
    parser.add_argument("--vel_yaw", type=float, default=0.0, help="Yaw velocity command (rad/s)")
    parser.add_argument("--vel_sweep", action="store_true",
                        help="Sweep vel_x from --vel_min to --vel_max, test each velocity level")
    parser.add_argument("--vel_min", type=float, default=0.0, help="Sweep start velocity (m/s)")
    parser.add_argument("--vel_max", type=float, default=1.0, help="Sweep end velocity (m/s)")
    parser.add_argument("--vel_step", type=float, default=0.1, help="Sweep velocity increment (m/s)")
    parser.add_argument("--steps_per_vel", type=int, default=200,
                        help="Steps per velocity level in sweep mode (default: 200 = 4s @50Hz)")
    parser.add_argument("--keyboard", action="store_true", help="Use keyboard for velocity commands")
    parser.add_argument("--num_steps", type=int, default=10000, help="Number of control steps")
    parser.add_argument("--terrain_difficulty", type=float, default=None,
                        help="Scale terrain height by this factor (0-1). Example: current p3_coarse curriculum is about 0.625.")
    parser.add_argument("--record", type=str, default=None, help="Record video to this path (EGL offscreen)")
    parser.add_argument("--terrain", type=str, default=None, help="Terrain type: 'p3' or 'p3b'")
    parser.add_argument("--phase", type=str, default=None,
                        help="Phase ID (p1/p2/p3/p3b/p4) — auto-selects terrain. Explicit --terrain takes priority.")
    parser.add_argument("--flat", action="store_true",
                        help="Force flat ground (overrides terrain from --phase)")
    parser.add_argument("--show_viewer", action="store_true", default=True, help="Show MuJoCo viewer")
    parser.add_argument("--csv", type=str, default=None, nargs='?', const='auto',
                        help="Log simulation data to CSV. --csv alone auto-names; --csv path.csv uses custom path")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# MuJoCo leg joint names in training order
# ---------------------------------------------------------------------------
LEG_JOINTS_MJCF = [
    "JOINT_HIP_PITCH_L", "JOINT_HIP_ROLL_L", "JOINT_HIP_YAW_L",
    "JOINT_KNEE_PITCH_L", "JOINT_ANKLE_PITCH_L", "JOINT_ANKLE_ROLL_L",
    "JOINT_HIP_PITCH_R", "JOINT_HIP_ROLL_R", "JOINT_HIP_YAW_R",
    "JOINT_KNEE_PITCH_R", "JOINT_ANKLE_PITCH_R", "JOINT_ANKLE_ROLL_R",
]

LEG_ACTUATOR_NAMES = [
    "left_hip_pitch_actuator", "left_hip_roll_actuator", "left_hip_yaw_actuator",
    "left_knee_actuator", "left_ankle_pitch_actuator", "left_ankle_roll_actuator",
    "right_hip_pitch_actuator", "right_hip_roll_actuator", "right_hip_yaw_actuator",
    "right_knee_actuator", "right_ankle_pitch_actuator", "right_ankle_roll_actuator",
]

# Default PD gains
DEFAULT_KP = np.array([
    100, 100, 100, 150, 60, 60,
    100, 100, 100, 150, 60, 60,
], dtype=np.float64)

DEFAULT_KD = np.array([
    4.0, 4.0, 4.0, 5.0, 3.0, 3.0,
    4.0, 4.0, 4.0, 5.0, 3.0, 3.0,
], dtype=np.float64)

DEFAULT_JOINT_POS = np.array([
    -0.35, 0.0, 0.0, 0.7, -0.35, 0.0,
    -0.35, 0.0, 0.0, 0.7, -0.35, 0.0,
], dtype=np.float64)

ACTION_SCALE = 0.25
OBS_SCALE_ANG_VEL = 0.2
OBS_SCALE_JOINT_VEL = 0.05
GAIT_PERIOD = 0.6
PHYSICS_DT = 0.002
DECIMATION = 10
CONTROL_DT = PHYSICS_DT * DECIMATION

CONTACT_FOOT_FRICTION = (0.65, 0.02, 0.02)
CONTACT_GROUND_FRICTION = (0.65, 0.02, 0.02)
CONTACT_FOOT_SOLREF = (-3000, -300)
CONTACT_GROUND_SOLREF = (-3000, -300)
CONTACT_FOOT_SOLIMP = (0.9, 0.99, 0.001, 0.5, 2)
CONTACT_GROUND_SOLIMP = (0.9, 0.99, 0.001, 0.5, 2)

OBS_DIM_PER_FRAME = 47
HISTORY_LENGTH = 5
OBS_DIM_TOTAL = OBS_DIM_PER_FRAME * HISTORY_LENGTH
TERM_DIMS = [3, 3, 3, 12, 12, 12, 2]
NUM_TERMS = len(TERM_DIMS)


class PolicyRunner:
    def __init__(self, policy_path, use_onnx=False):
        self.use_onnx = use_onnx
        if use_onnx:
            import onnxruntime as ort
            self.session = ort.InferenceSession(policy_path)
        else:
            import torch
            self.model = torch.jit.load(policy_path, map_location="cpu")
            self.model.eval()

    def predict(self, obs: np.ndarray) -> np.ndarray:
        if self.use_onnx:
            obs_input = obs.astype(np.float32).reshape(1, -1)
            return self.session.run(None, {"obs": obs_input})[0].flatten()
        else:
            import torch
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).float().reshape(1, -1)
                return self.model(obs_t).numpy().flatten()


class ObservationBuffer:
    def __init__(self):
        self.buffer = np.zeros((HISTORY_LENGTH, OBS_DIM_PER_FRAME))

    def reset(self, initial_obs: np.ndarray):
        for i in range(HISTORY_LENGTH):
            self.buffer[i] = initial_obs

    def append(self, obs: np.ndarray) -> np.ndarray:
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = obs
        return self.get()

    def get(self) -> np.ndarray:
        result = np.empty(OBS_DIM_TOTAL, dtype=np.float64)
        src_col = 0
        dst = 0
        for term_dim in TERM_DIMS:
            for h in range(HISTORY_LENGTH):
                result[dst:dst + term_dim] = self.buffer[h, src_col:src_col + term_dim]
                dst += term_dim
            src_col += term_dim
        return result


class KeyboardController:
    # GLFW key codes (used by MuJoCo viewer)
    # Arrow keys avoid conflict with MuJoCo viewer's built-in WASD camera/lighting controls
    GLFW_KEYS = {
        265: 'up',       # GLFW_KEY_UP
        264: 'down',     # GLFW_KEY_DOWN
        263: 'left',     # GLFW_KEY_LEFT
        262: 'right',    # GLFW_KEY_RIGHT
        81: 'q',         # GLFW_KEY_Q
        69: 'e',         # GLFW_KEY_E
        32: 'space',     # GLFW_KEY_SPACE
        256: 'esc',      # GLFW_KEY_ESCAPE
        91: 'lbracket',  # GLFW_KEY_LEFT_BRACKET  [
        93: 'rbracket',  # GLFW_KEY_RIGHT_BRACKET ]
        87: 'w',         # GLFW_KEY_W
        83: 's',         # GLFW_KEY_S
        258: 'tab',      # GLFW_KEY_TAB
    }

    # Joint short names for display
    JOINT_NAMES = [
        "hip_p_l", "hip_r_l", "hip_y_l", "knee_p_l", "ank_p_l", "ank_r_l",
        "hip_p_r", "hip_r_r", "hip_y_r", "knee_p_r", "ank_p_r", "ank_r_r",
    ]

    def __init__(self, initial_vel=None, manual_mode=False, default_joint_pos=None):
        self.manual_mode = manual_mode
        self.vel_cmd = np.array(initial_vel, dtype=np.float64) if initial_vel is not None else np.array([0.0, 0.0, 0.0])
        self.lin_vel_range = (-0.5, 1.0)
        self.ang_vel_range = (-0.5, 0.5)
        self.vel_step = 0.1
        self.running = True
        self.viewer_active = False  # set True when MuJoCo viewer callback is registered
        self.panel_visible = True

        # Manual mode state
        if manual_mode:
            self.selected_joint = 0
            self.joint_targets = (default_joint_pos if default_joint_pos is not None else DEFAULT_JOINT_POS).copy()
            print("[MANUAL] Direct joint control mode (no policy)")
            print("  [/]=select joint  Up/Down=±0.1  W/S=±0.02  Space=reset joints  R=reset robot  Tab=panel  Esc=quit")
        else:
            print("Keyboard control: Up/Down=forward/back, Left/Right=turn, Q/E=lateral, Space=stop, Esc=quit")

    def handle_viewer_key(self, keycode):
        """Callback for MuJoCo viewer key presses (GLFW keycodes)."""
        key = self.GLFW_KEYS.get(keycode)
        if key:
            self._apply_key(key)

    def update(self):
        """Poll terminal for key presses (headless / no viewer mode)."""
        if self.viewer_active:
            return
        try:
            import msvcrt
            if msvcrt.kbhit():
                key = msvcrt.getch()
                self._apply_msvcrt(key)
        except ImportError:
            import sys, select
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.readline().strip()
                self._apply_key(key)

    def _apply_msvcrt(self, key):
        """Convert msvcrt byte to key name and apply."""
        mapping = {b'w': 'w' if self.manual_mode else 'up',
                   b's': 's' if self.manual_mode else 'down',
                   b'a': 'left', b'd': 'right',
                   b'q': 'q', b'e': 'e', b' ': 'space', b'\x1b': 'esc',
                   b'r': 'r',
                   b'H': 'up', b'P': 'down', b'K': 'left', b'M': 'right',
                   b'[': 'lbracket', b']': 'rbracket'}
        key_name = mapping.get(key)
        if key_name:
            self._apply_key(key_name)

    def _apply_key(self, key):
        """Shared key action handler."""
        if self.manual_mode:
            self._apply_key_manual(key)
        else:
            self._apply_key_policy(key)

    def _apply_key_policy(self, key):
        """Key handler for policy (velocity command) mode."""
        if key == 'up':
            self.vel_cmd[0] = min(self.vel_cmd[0] + self.vel_step, self.lin_vel_range[1])
            self._print_vel()
        elif key == 'down':
            self.vel_cmd[0] = max(self.vel_cmd[0] - self.vel_step, self.lin_vel_range[0])
            self._print_vel()
        elif key == 'left':
            self.vel_cmd[2] = max(self.vel_cmd[2] - self.vel_step, self.ang_vel_range[0])
            self._print_vel()
        elif key == 'right':
            self.vel_cmd[2] = min(self.vel_cmd[2] + self.vel_step, self.ang_vel_range[1])
            self._print_vel()
        elif key == 'q':
            self.vel_cmd[1] = min(self.vel_cmd[1] + self.vel_step, 0.5)
            self._print_vel()
        elif key == 'e':
            self.vel_cmd[1] = max(self.vel_cmd[1] - self.vel_step, -0.5)
            self._print_vel()
        elif key == 'space':
            self.vel_cmd[:] = 0.0
            print("  [CMD] Stop (vel=0)")
        elif key == 'esc':
            self.running = False

    def _apply_key_manual(self, key):
        """Key handler for manual joint control mode."""
        if key == 'lbracket':
            self.selected_joint = (self.selected_joint - 1) % 12
            self._print_manual_status()
        elif key == 'rbracket':
            self.selected_joint = (self.selected_joint + 1) % 12
            self._print_manual_status()
        elif key == 'up':
            self.joint_targets[self.selected_joint] += 0.1
            self._print_manual_status()
        elif key == 'down':
            self.joint_targets[self.selected_joint] -= 0.1
            self._print_manual_status()
        elif key == 'w':
            self.joint_targets[self.selected_joint] += 0.02
            self._print_manual_status()
        elif key == 's':
            self.joint_targets[self.selected_joint] -= 0.02
            self._print_manual_status()
        elif key == 'space':
            self.joint_targets[:] = DEFAULT_JOINT_POS
            print("  [MANUAL] All joints reset to default")
        elif key == 'r':
            self._reset_requested = True
            print("  [MANUAL] Robot reset requested")
        elif key == 'tab':
            self.panel_visible = not self.panel_visible
            state = "shown" if self.panel_visible else "hidden"
            print(f"  [MANUAL] Left panel: {state}")
        elif key == 'esc':
            self.running = False

    def _print_vel(self):
        print(f"  [CMD] vel=({self.vel_cmd[0]:.1f}, {self.vel_cmd[1]:.1f}, {self.vel_cmd[2]:.1f})")

    def get_command(self):
        return self.vel_cmd.copy()

    def get_joint_targets(self):
        """Return current manual joint targets (manual mode only)."""
        return self.joint_targets.copy()

    def _print_manual_status(self):
        j = self.selected_joint
        print(f"  [{j:2d}] {self.JOINT_NAMES[j]:10s} target={self.joint_targets[j]:+.3f} rad")


def quat_to_rot_matrix(quat_wxyz):
    w, x, y, z = quat_wxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ])


def compute_gait_phase(sim_time, vel_cmd, period=GAIT_PERIOD):
    cmd_norm = np.linalg.norm(vel_cmd)
    if cmd_norm < 0.02:
        return np.array([1.0, 1.0])
    phase = (sim_time % period) / period
    sin_pos = math.sin(2.0 * math.pi * phase)
    left_stance = 1.0 if sin_pos >= 0 else 0.0
    right_stance = 1.0 if sin_pos < 0 else 0.0
    return np.array([left_stance, right_stance])


class MuJoCoDeploy:
    def __init__(self, mjcf_path, policy_runner=None, deploy_cfg=None, vel_cmd=None, terrain=None, terrain_difficulty=1.0):
        import mujoco

        self.policy = policy_runner
        self.manual_mode = policy_runner is None
        self.vel_cmd = np.array(vel_cmd) if vel_cmd is not None else np.array([0.5, 0.0, 0.0])

        # PD gains: load from deploy.yaml (training uses IdealPDActuator = explicit PD,
        # same as MuJoCo, so values match directly — no sim2sim boost needed)
        # default_joint_pos: deploy.yaml exports in wrong joint order, use hardcoded
        self.default_joint_pos = DEFAULT_JOINT_POS.copy()

        if deploy_cfg:
            self.kp = np.array(deploy_cfg.get("stiffness", DEFAULT_KP.tolist()))
            self.kd = np.array(deploy_cfg.get("damping", DEFAULT_KD.tolist()))
            action_cfg = deploy_cfg.get("actions", {}).get("JointPositionAction", {})
            self.action_scale = np.array(action_cfg.get("scale", [ACTION_SCALE] * 12))
            self.action_offset = np.array(action_cfg.get("offset", DEFAULT_JOINT_POS.tolist()))
        else:
            self.kp = DEFAULT_KP.copy()
            self.kd = DEFAULT_KD.copy()
            self.action_scale = np.full(12, ACTION_SCALE)
            self.action_offset = DEFAULT_JOINT_POS.copy()

        # Load model (with optional terrain)
        if terrain:
            self.model = load_model_with_terrain(mjcf_path, terrain, difficulty=terrain_difficulty)
        else:
            self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.model.opt.timestep = PHYSICS_DT
        self.data = mujoco.MjData(self.model)

        # Resolve joint and actuator indices
        self.leg_joint_ids = []
        for name in LEG_JOINTS_MJCF:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert jid >= 0, f"Joint '{name}' not found in MJCF model"
            self.leg_joint_ids.append(jid)

        self.leg_actuator_ids = []
        for name in LEG_ACTUATOR_NAMES:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            assert aid >= 0, f"Actuator '{name}' not found in MJCF model"
            self.leg_actuator_ids.append(aid)

        self.leg_qpos_addr = [self.model.jnt_qposadr[jid] for jid in self.leg_joint_ids]
        self.leg_dof_addr = [self.model.jnt_dofadr[jid] for jid in self.leg_joint_ids]

        for jid in self.leg_joint_ids:
            self.model.dof_damping[jid] = 0.0

        armature = np.array([0.02863, 0.02863, 0.02863, 0.02863, 0.01503, 0.01503,
                             0.02863, 0.02863, 0.02863, 0.02863, 0.01503, 0.01503])
        for i, jid in enumerate(self.leg_joint_ids):
            self.model.dof_armature[jid] = armature[i]

        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if name in ("l_foot", "r_foot"):
                self.model.geom_friction[geom_id] = CONTACT_FOOT_FRICTION
                self.model.geom_solref[geom_id] = CONTACT_FOOT_SOLREF
                self.model.geom_solimp[geom_id] = CONTACT_FOOT_SOLIMP
            elif name == "ground":
                self.model.geom_friction[geom_id] = CONTACT_GROUND_FRICTION
                self.model.geom_solref[geom_id] = CONTACT_GROUND_SOLREF
                self.model.geom_solimp[geom_id] = CONTACT_GROUND_SOLIMP

        self.effort_limits = np.array([120, 120, 120, 120, 50, 50,
                                       120, 120, 120, 120, 50, 50], dtype=np.float64)

        self._initial_vel_cmd = self.vel_cmd.copy()
        self.obs_buffer = ObservationBuffer()
        self.last_action = np.zeros(12)
        self.last_torques = np.zeros(12)
        self.sim_time = 0.0
        self.reset()

    def reset(self):
        import mujoco
        self.vel_cmd[:] = self._initial_vel_cmd
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.69
        self.data.qpos[3] = 1.0
        self.data.qpos[4:7] = 0.0
        for i, addr in enumerate(self.leg_qpos_addr):
            self.data.qpos[addr] = self.default_joint_pos[i]
        mujoco.mj_forward(self.model, self.data)
        self.last_action = np.zeros(12)
        self.sim_time = 0.0
        obs = self._build_obs_frame()
        self.obs_buffer.reset(obs)
        for _ in range(HISTORY_LENGTH):
            tgt = self.default_joint_pos
            for _ in range(DECIMATION):
                cp = np.array([self.data.qpos[a] for a in self.leg_qpos_addr])
                cv = np.array([self.data.qvel[a] for a in self.leg_dof_addr])
                torques = np.clip(self.kp * (tgt - cp) - self.kd * cv, -self.effort_limits, self.effort_limits)
                for i, act_id in enumerate(self.leg_actuator_ids):
                    self.data.ctrl[act_id] = torques[i]
                mujoco.mj_step(self.model, self.data)
            self.sim_time += CONTROL_DT
            obs = self._build_obs_frame()
            self.obs_buffer.append(obs)

    def _get_pelvis_rot(self):
        quat = self.data.qpos[3:7].copy()
        return quat_to_rot_matrix(quat)

    def _build_obs_frame(self):
        obs = np.zeros(OBS_DIM_PER_FRAME)
        idx = 0
        R_wb = self._get_pelvis_rot()
        omega_world = self.data.qvel[3:6].copy()
        omega_body = R_wb @ omega_world
        obs[idx:idx+3] = omega_body * OBS_SCALE_ANG_VEL
        idx += 3
        gravity_world = np.array([0.0, 0.0, -1.0])
        gravity_body = R_wb @ gravity_world
        obs[idx:idx+3] = gravity_body
        idx += 3
        obs[idx:idx+3] = self.vel_cmd
        idx += 3
        joint_pos = np.array([self.data.qpos[a] for a in self.leg_qpos_addr])
        obs[idx:idx+12] = joint_pos - self.default_joint_pos
        idx += 12
        joint_vel = np.array([self.data.qvel[a] for a in self.leg_dof_addr])
        obs[idx:idx+12] = joint_vel * OBS_SCALE_JOINT_VEL
        idx += 12
        obs[idx:idx+12] = self.last_action
        idx += 12
        obs[idx:idx+2] = compute_gait_phase(self.sim_time, self.vel_cmd)
        idx += 2
        return obs

    def step(self):
        import mujoco
        obs_history = self.obs_buffer.get()
        action = self.policy.predict(obs_history)
        action = np.clip(action, -100.0, 100.0)
        self.last_action = action.copy()
        target_pos = self.action_offset + action * self.action_scale
        for _ in range(DECIMATION):
            current_pos = np.array([self.data.qpos[a] for a in self.leg_qpos_addr])
            current_vel = np.array([self.data.qvel[a] for a in self.leg_dof_addr])
            torques = np.clip(
                self.kp * (target_pos - current_pos) - self.kd * current_vel,
                -self.effort_limits, self.effort_limits
            )
            for i, act_id in enumerate(self.leg_actuator_ids):
                self.data.ctrl[act_id] = torques[i]
            mujoco.mj_step(self.model, self.data)
        self.last_torques = torques.copy()
        self.sim_time += CONTROL_DT
        if self.data.qpos[2] < 0.3 or (self.data.qpos[4]**2 + self.data.qpos[5]**2 + self.data.qpos[6]**2) > 0.5:
            return True
        new_obs = self._build_obs_frame()
        self.obs_buffer.append(new_obs)
        return False

    def step_manual(self, joint_targets):
        """PD control step using externally provided joint targets (manual mode)."""
        import mujoco
        for _ in range(DECIMATION):
            current_pos = np.array([self.data.qpos[a] for a in self.leg_qpos_addr])
            current_vel = np.array([self.data.qvel[a] for a in self.leg_dof_addr])
            torques = np.clip(
                self.kp * (joint_targets - current_pos) - self.kd * current_vel,
                -self.effort_limits, self.effort_limits
            )
            for i, act_id in enumerate(self.leg_actuator_ids):
                self.data.ctrl[act_id] = torques[i]
            mujoco.mj_step(self.model, self.data)
        self.last_torques = torques.copy()
        self.last_action = np.zeros(12)
        self.sim_time += CONTROL_DT
        if self.data.qpos[2] < 0.3 or (self.data.qpos[4]**2 + self.data.qpos[5]**2 + self.data.qpos[6]**2) > 0.5:
            return True
        return False

    def reset_manual(self):
        """Reset robot pose for manual mode with PD warmup to stabilize standing."""
        import mujoco
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.69
        self.data.qpos[3] = 1.0
        self.data.qpos[4:7] = 0.0
        for i, addr in enumerate(self.leg_qpos_addr):
            self.data.qpos[addr] = self.default_joint_pos[i]
        mujoco.mj_forward(self.model, self.data)
        self.last_action = np.zeros(12)
        self.last_torques = np.zeros(12)
        self.sim_time = 0.0
        # PD warmup — let the robot settle before returning control to user
        for _ in range(50):
            tgt = self.default_joint_pos
            for _ in range(DECIMATION):
                cp = np.array([self.data.qpos[a] for a in self.leg_qpos_addr])
                cv = np.array([self.data.qvel[a] for a in self.leg_dof_addr])
                torques = np.clip(self.kp * (tgt - cp) - self.kd * cv, -self.effort_limits, self.effort_limits)
                for i, act_id in enumerate(self.leg_actuator_ids):
                    self.data.ctrl[act_id] = torques[i]
                mujoco.mj_step(self.model, self.data)
            self.sim_time += CONTROL_DT

    def get_robot_state(self):
        pos = self.data.qpos[:3].copy()
        return {"x": pos[0], "y": pos[1], "z": pos[2], "time": self.sim_time}


def main():
    args = parse_args()

    if not args.manual and args.policy is None:
        print("[ERROR] --policy is required when not using --manual mode")
        return

    deploy_cfg = None
    if args.deploy_cfg:
        with open(args.deploy_cfg, "r") as f:
            deploy_cfg = yaml.unsafe_load(f)
        print(f"[INFO] Loaded deploy config from {args.deploy_cfg}")

    # Policy loading (skip in manual mode)
    policy = None
    if not args.manual:
        policy = PolicyRunner(args.policy, use_onnx=args.onnx)
        print(f"[INFO] Loaded policy from {args.policy} (ONNX={args.onnx})")

    vel_cmd = [args.vel_x, args.vel_y, args.vel_yaw]

    # Velocity sweep setup
    vel_sweep = args.vel_sweep
    vel_stable_max = None
    sweep_results = []  # per-velocity results

    if vel_sweep and not (args.keyboard or args.manual):
        vel_levels = list(np.arange(args.vel_min, args.vel_max + args.vel_step / 2, args.vel_step))
        steps_per_vel = args.steps_per_vel
        total_sweep_steps = len(vel_levels) * steps_per_vel
        vel_cmd[0] = vel_levels[0]
        print(f"[INFO] Velocity sweep: {len(vel_levels)} levels × {steps_per_vel} steps "
              f"({vel_levels[0]:.1f} → {vel_levels[-1]:.1f} m/s, "
              f"{total_sweep_steps} total steps = {total_sweep_steps * CONTROL_DT:.0f}s)")
    else:
        vel_levels = None
        steps_per_vel = None
        total_sweep_steps = None
        vel_cmd[0] = args.vel_x

    kb_controller = None
    if args.keyboard or args.manual:
        kb_controller = KeyboardController(
            initial_vel=vel_cmd,
            manual_mode=args.manual,
            default_joint_pos=DEFAULT_JOINT_POS,
        )

    # Phase-aware terrain selection: --flat > --terrain > --phase
    if args.flat:
        terrain = None
        if args.phase is not None:
            print(f"[INFO] --flat forced: ignoring terrain from phase '{args.phase}'")
    else:
        terrain = args.terrain
        if terrain is None and args.phase is not None:
            terrain = PHASE_TERRAIN.get(args.phase)
            if terrain is None:
                print(f"[INFO] Phase '{args.phase}' -> flat ground (no terrain needed)")
            else:
                print(f"[INFO] Phase '{args.phase}' -> auto-selected terrain '{terrain}'")
        elif terrain is not None and args.phase is not None:
            print(f"[INFO] Phase '{args.phase}' overridden by explicit --terrain '{terrain}'")

    terrain_diff = args.terrain_difficulty if args.terrain_difficulty is not None else 1.0
    if terrain == "p3" and args.terrain_difficulty is None:
        print("[WARN] --terrain_difficulty not set; using full p3 height (1.0). "
              "For the current p3_coarse run, ~0.625 is closer to the trained curriculum.")
    env = MuJoCoDeploy(args.mjcf, policy, deploy_cfg=deploy_cfg, vel_cmd=vel_cmd, terrain=terrain,
                        terrain_difficulty=terrain_diff)
    print(f"[INFO] MuJoCo model loaded from {args.mjcf}" +
          (f" with terrain={terrain}" if terrain else " (flat ground)"))
    print(f"[INFO] Control frequency: {1.0/CONTROL_DT:.0f}Hz, Physics: {1.0/PHYSICS_DT:.0f}Hz")
    if not args.manual:
        print(f"[INFO] Observation: {OBS_DIM_TOTAL}d ({OBS_DIM_PER_FRAME}d x {HISTORY_LENGTH} frames)")
        print(f"[INFO] Velocity command: {vel_cmd}")

    # CSV data logger
    csv_file = None
    csv_writer = None
    csv_path = None
    LOG_JOINT_NAMES_SHORT = [
        "hip_p_l", "hip_r_l", "hip_y_l", "knee_p_l", "ank_p_l", "ank_r_l",
        "hip_p_r", "hip_r_r", "hip_y_r", "knee_p_r", "ank_p_r", "ank_r_r",
    ]
    if args.csv:
        import csv
        if args.csv == 'auto':
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            phase_tag = args.phase or "test"
            csv_path = f"logs/p/{phase_tag}/{timestamp}/{phase_tag}.csv"
        else:
            csv_path = args.csv
        log_dir = os.path.dirname(csv_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        csv_file = open(csv_path, 'w', newline='')
        csv_header = ['step', 'time', 'x', 'y', 'z']
        csv_header += [f'qpos_{n}' for n in LOG_JOINT_NAMES_SHORT]
        csv_header += [f'qvel_{n}' for n in LOG_JOINT_NAMES_SHORT]
        csv_header += [f'tau_{n}' for n in LOG_JOINT_NAMES_SHORT]
        csv_header += [f'action_{n}' for n in LOG_JOINT_NAMES_SHORT]
        csv_header += ['cmd_vx', 'cmd_vy', 'cmd_vyaw', 'fall']
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_header)
        print(f"[INFO] Logging data to {csv_path}")

    import mujoco
    from mujoco import viewer as mujoco_viewer
    viewer = None
    renderer = None
    frames = []

    if args.record:
        renderer = mujoco.Renderer(env.model, height=480, width=640)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(env.model, cam)
        cam.distance = 3.0
        cam.elevation = -20
        cam.azimuth = 90
        print(f"[INFO] EGL offscreen recording -> {args.record}")
        args.show_viewer = False
    elif args.show_viewer:
        try:
            if kb_controller:
                viewer = mujoco_viewer.launch_passive(
                    env.model, env.data,
                    key_callback=kb_controller.handle_viewer_key,
                )
                kb_controller.viewer_active = True
            else:
                viewer = mujoco_viewer.launch_passive(env.model, env.data)
            print("[INFO] Viewer launched (Tab=toggle left panel, close window to end)")
        except Exception as e:
            print(f"[WARNING] Could not launch MuJoCo viewer: {e}")
            viewer = None

    fall_count = 0
    num_total = total_sweep_steps if vel_sweep and vel_levels else args.num_steps
    print(f"[INFO] Running {num_total} steps...")
    try:
        for step in range(num_total):

            # ── Sweep: velocity transition ──
            if vel_sweep and vel_levels:
                vel_idx = step // steps_per_vel
                local_step = step % steps_per_vel
                if local_step == 0:
                    # Start of a new velocity level
                    env.vel_cmd[0] = vel_levels[vel_idx]
                    env.reset()
                    current_vel_x_start = env.get_robot_state()['x']
                    current_vel_falls = 0
            if kb_controller:
                kb_controller.update()
                if not kb_controller.running:
                    break

                if args.manual:
                    # Check for reset request
                    if getattr(kb_controller, '_reset_requested', False):
                        kb_controller._reset_requested = False
                        env.reset_manual()
                        kb_controller.joint_targets[:] = DEFAULT_JOINT_POS
                        fall_count = 0
                        continue

                    joint_targets = kb_controller.get_joint_targets()
                    fell = env.step_manual(joint_targets)
                else:
                    env.vel_cmd = kb_controller.get_command()
                    fell = env.step()

                if fell:
                    if args.manual:
                        env.reset_manual()
                        kb_controller.joint_targets[:] = DEFAULT_JOINT_POS
                    else:
                        env.reset()
                    fall_count += 1
            else:
                fell = env.step()
                if fell:
                    if vel_sweep and vel_levels:
                        env.reset()
                        current_vel_falls += 1
                    else:
                        env.reset()
                    fall_count += 1

            if csv_writer:
                state = env.get_robot_state()
                qpos = [env.data.qpos[a] for a in env.leg_qpos_addr]
                qvel = [env.data.qvel[a] for a in env.leg_dof_addr]
                row = [step, f"{state['time']:.4f}",
                       f"{state['x']:.4f}", f"{state['y']:.4f}", f"{state['z']:.4f}"]
                row += [f"{v:.4f}" for v in qpos]
                row += [f"{v:.4f}" for v in qvel]
                row += [f"{v:.2f}" for v in env.last_torques]
                row += [f"{v:.4f}" for v in env.last_action]
                row += [f"{env.vel_cmd[0]:.2f}", f"{env.vel_cmd[1]:.2f}", f"{env.vel_cmd[2]:.2f}",
                        1 if fell else 0]
                csv_writer.writerow(row)

            if renderer:
                renderer.update_scene(env.data, camera=cam)
                frames.append(renderer.render().copy())

            if viewer is not None:
                try:
                    viewer.sync()
                except Exception:
                    break

            if step % 500 == 0:
                state = env.get_robot_state()
                if args.manual:
                    j = kb_controller.selected_joint if kb_controller else 0
                    print(f"  Step {step:5d} | t={state['time']:.1f}s | "
                          f"pos=({state['x']:.2f}, {state['y']:.2f}, {state['z']:.2f}) | "
                          f"[{j}] {KeyboardController.JOINT_NAMES[j]}={kb_controller.joint_targets[j]:+.3f} | "
                          f"falls={fall_count}")
                else:
                    vel_cmd_str = (f"cmd=({env.vel_cmd[0]:.1f}, {env.vel_cmd[1]:.1f}, {env.vel_cmd[2]:.1f})"
                                   if kb_controller else "")
                    print(f"  Step {step:5d} | t={state['time']:.1f}s | "
                          f"pos=({state['x']:.2f}, {state['y']:.2f}, {state['z']:.2f}) | "
                          f"{vel_cmd_str} | falls={fall_count}")

            # ── Sweep: end of velocity level ──
            if vel_sweep and vel_levels:
                vel_idx = step // steps_per_vel
                local_step = step % steps_per_vel
                if local_step == steps_per_vel - 1:
                    # Last step of this velocity level — record results
                    state = env.get_robot_state()
                    x_delta = state['x'] - current_vel_x_start
                    vel = vel_levels[vel_idx]
                    sweep_results.append({
                        "velocity": round(vel, 2),
                        "falls": current_vel_falls,
                        "x_displacement": round(x_delta, 3),
                    })
                    tag = "STABLE" if current_vel_falls == 0 else f"{current_vel_falls} fall{'s' if current_vel_falls > 1 else ''}"
                    print(f"  [SWEEP] {vel:.1f} m/s: {tag}, Δx={x_delta:+.2f}m")

            if viewer and not args.record:
                time.sleep(max(0, CONTROL_DT - 0.001))

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    if args.record and frames:
        import imageio
        print(f"[INFO] Saving {len(frames)} frames to {args.record}...")
        imageio.mimwrite(args.record, frames, fps=int(1.0 / CONTROL_DT))
        sz = os.path.getsize(args.record) / (1024 * 1024)
        print(f"[INFO] Done! {args.record} ({sz:.1f} MB), Falls: {fall_count}")

        # Save sweep metadata alongside video
        if vel_sweep and sweep_results:
            import json
            meta_path = args.record.rsplit('.', 1)[0] + '_sweep.json'
            stable_max = max(
                (r["velocity"] for r in sweep_results if r["falls"] == 0),
                default=None,
            )
            sweep_metadata = {
                "vel_min": args.vel_min,
                "vel_max": args.vel_max,
                "vel_step": args.vel_step,
                "steps_per_vel": steps_per_vel,
                "vel_stable_max": stable_max,
                "total_falls": fall_count,
                "total_steps": len(frames),
                "per_velocity": sweep_results,
            }
            with open(meta_path, 'w') as f:
                json.dump(sweep_metadata, f, indent=2)
            print(f"[INFO] Sweep summary: stable_max={stable_max} m/s, "
                  f"total_falls={fall_count}")
            print(f"[INFO] Saved: {meta_path}")

    if viewer is not None:
        try:
            viewer.close()
        except Exception:
            pass

    if csv_file:
        csv_file.close()
        print(f"[INFO] Log saved: {csv_path} ({step+1} rows)")
        print(f"[ANALYZE] python -c \"import pandas as pd; print(pd.read_csv('{csv_path}').describe())\"")

    print("[INFO] Simulation ended")


if __name__ == "__main__":
    main()
