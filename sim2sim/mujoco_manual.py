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

    # Phase-aware terrain (auto-selects the latest verified RTX artifact baseline):
    python sim2sim/mujoco_manual.py \
        --mjcf ../magicbot-z1_description/mjcf/MAGICBOTZ1.xml \
        --policy policy.pt --deploy_cfg deploy.yaml \
        --phase p3_fine --keyboard

    # Explicit terrain override:
    python sim2sim/mujoco_manual.py \
        --mjcf ../magicbot-z1_description/mjcf/MAGICBOTZ1.xml \
        --policy policy.pt --deploy_cfg deploy.yaml \
        --terrain p3_coarse --record /tmp/output.mp4

    # Manual joint control (no policy):
    python sim2sim/mujoco_manual.py \
        --mjcf ../magicbot-z1_description/mjcf/MAGICBOTZ1.xml \
        --manual --keyboard
"""

import argparse
import ctypes
import math
import os
import time
from pathlib import Path

# Set EGL for offscreen rendering on Linux (Windows uses default wgl)
if os.name != "nt":
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import yaml

# Phase → terrain mapping for auto terrain selection
# RTX run artifacts are the baseline for local play / recording.
# Flat phases keep terrain_type=None. Terrain-bearing phases map to the exact
# sub-phase snapshot that actually trained on RTX.
RTX_ARTIFACT_BASELINES = {
    "p1_coarse": {"terrain_type": None, "description": "RTX artifact 2026-05-06_15-47-12_p1_coarse"},
    "p1_fine": {"terrain_type": None, "description": "RTX artifact 2026-05-06_17-40-13_p1_fine"},
    "p2_coarse": {"terrain_type": None, "description": "RTX artifact 2026-05-15_17-44-46_p2_coarse"},
    "p2_fine": {"terrain_type": None, "description": "RTX artifact 2026-05-15_19-58-42_p2_fine"},
    "p3_coarse": {
        "terrain_type": "p3",
        "description": "RTX artifact 2026-05-15_21-55-08_p3_coarse",
        "terrain_length": 24.0,
        "terrain_width": 8.0,
        "horizontal_scale": 0.1,
        "grid_width": 0.6,
        "max_elev": 0.25,
        "random_grid_height_range": (0.0, 0.25),
        "layout": [("flat", 0.35), ("random_grid", 0.15), ("flat", 0.35), ("random_grid", 0.15)],
    },
    "p3_fine": {
        "terrain_type": "p3",
        "description": "RTX artifact 2026-05-16_12-13-26_p3_fine",
        "terrain_length": 24.0,
        "terrain_width": 8.0,
        "horizontal_scale": 0.1,
        "grid_width": 0.6,
        "max_elev": 0.35,
        "random_grid_height_range": (0.0, 0.35),
        "layout": [("flat", 0.35), ("random_grid", 0.15), ("flat", 0.35), ("random_grid", 0.15)],
    },
}

PHASE_ARTIFACT_BASELINES = {
    "p1": "p1_fine",
    "p1_coarse": "p1_coarse",
    "p1_fine": "p1_fine",
    "p2": "p2_fine",
    "p2_coarse": "p2_coarse",
    "p2_fine": "p2_fine",
    "p3": "p3_fine",
    "p3_coarse": "p3_coarse",
    "p3_fine": "p3_fine",
}

TERRAIN_ARTIFACT_ALIASES = {
    "p3": "p3_fine",
    "p3b": "p3_fine",
}


def resolve_artifact_baseline(phase=None, terrain=None, flat=False):
    if flat:
        return None

    requested = terrain if terrain is not None else phase
    if requested is None:
        return None

    if requested in TERRAIN_ARTIFACT_ALIASES:
        mapped = TERRAIN_ARTIFACT_ALIASES[requested]
        if requested == "p3b":
            print(f"[WARN] Legacy selector '{requested}' is deprecated; using RTX artifact baseline '{mapped}'")
        requested = mapped
    elif requested in PHASE_ARTIFACT_BASELINES:
        requested = PHASE_ARTIFACT_BASELINES[requested]

    if requested not in RTX_ARTIFACT_BASELINES:
        raise ValueError(
            f"No RTX artifact baseline defined for '{requested}'. "
            "Use one of: p1_coarse, p1_fine, p2_coarse, p2_fine, p3_coarse, p3_fine, or phase aliases p1/p2/p3."
        )
    return requested


# ---------------------------------------------------------------------------
# Terrain generation (matching RTX training artifacts)
# ---------------------------------------------------------------------------
def generate_terrain_data(artifact_key="p3_fine", seed=42, difficulty=1.0):
    """Generate terrain heightmap from an RTX artifact baseline."""
    profile = RTX_ARTIFACT_BASELINES[artifact_key]
    terrain_type = profile["terrain_type"]
    if terrain_type is None:
        raise ValueError(f"Artifact baseline '{artifact_key}' is flat and does not need hfield generation")

    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    H_SCALE = profile["horizontal_scale"]
    TERRAIN_L = profile["terrain_length"]
    TERRAIN_W = profile["terrain_width"]
    MAX_ELEV = profile["max_elev"]
    GRID_WIDTH = profile["grid_width"]
    RANDOM_GRID_HEIGHT_RANGE = profile["random_grid_height_range"]
    LAYOUT = profile["layout"]

    ncol = int(TERRAIN_L / H_SCALE)
    nrow = int(TERRAIN_W / H_SCALE)
    rng = np.random.default_rng(seed)
    hmap = np.zeros((nrow, ncol), dtype=np.float64)

    gw = max(2, int(round(GRID_WIDTH / H_SCALE)))
    protrusion_coverage = float(sum(proportion for stype, proportion in LAYOUT if stype != "flat"))
    protrusion_coverage = float(np.clip(protrusion_coverage, 0.0, 1.0))
    target_active_cells = int(round(nrow * ncol * protrusion_coverage))

    # Use sparse smooth bumps with compact support so heights vary continuously
    # from 0 to the sampled peak instead of forming flat step-like blocks.
    rr, cc = np.meshgrid(np.arange(nrow), np.arange(ncol), indexing="ij")
    avg_radius = max(2.0, gw * 0.55)
    bump_area = math.pi * avg_radius * avg_radius
    num_bumps = max(1, int(round(target_active_cells / max(1.0, bump_area))))
    num_bumps = min(num_bumps, max(1, nrow * ncol // 16))

    for _ in range(num_bumps):
        center_r = rng.integers(0, nrow)
        center_c = rng.integers(0, ncol)
        radius = rng.uniform(gw * 0.4, gw * 0.85)
        peak = rng.uniform(RANDOM_GRID_HEIGHT_RANGE[0], RANDOM_GRID_HEIGHT_RANGE[1]) / MAX_ELEV

        dist = np.sqrt((rr - center_r) ** 2 + (cc - center_c) ** 2)
        inside = dist <= radius
        if not np.any(inside):
            continue

        # Raised-cosine bump: exactly 0 outside support, smooth inside.
        local = 0.5 * (1.0 + np.cos(np.pi * dist[inside] / radius))
        bump = np.zeros_like(hmap)
        bump[inside] = peak * local
        hmap = np.maximum(hmap, bump)

    if difficulty < 1.0:
        hmap *= difficulty
        effective_max = MAX_ELEV * difficulty
        print(f"[TERRAIN] Extra difficulty scale={difficulty:.3f}: scaling height to {effective_max:.3f}m "
              f"(artifact max {MAX_ELEV:.3f}m)")
    else:
        effective_max = MAX_ELEV

    print(f"[TERRAIN] Generated artifact '{artifact_key}' ({terrain_type}): {TERRAIN_L}m x {TERRAIN_W}m, "
          f"grid {nrow}x{ncol}, protrusion_coverage={protrusion_coverage:.2f}, "
          f"elevation [{hmap.min() * MAX_ELEV:.3f}, {hmap.max() * MAX_ELEV:.3f}]m, "
          f"effective_max={effective_max:.3f}m")
    return nrow, ncol, TERRAIN_L / 2, TERRAIN_W / 2, effective_max, hmap


def load_model_with_terrain(mjcf_path, artifact_key, difficulty=1.0):
    """Load MuJoCo model with terrain hfield injected via Python API."""
    import mujoco

    nrow, ncol, half_x, half_y, max_elev, terrain_data = generate_terrain_data(artifact_key, difficulty=difficulty)

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
    parser.add_argument("--vel_x", type=float, default=0.3, help="Forward velocity command (m/s)")
    parser.add_argument("--vel_y", type=float, default=0.0, help="Lateral velocity command (m/s)")
    parser.add_argument("--vel_yaw", type=float, default=0.0, help="Yaw velocity command (rad/s)")
    parser.add_argument("--keyboard", action="store_true", help="Use keyboard for velocity commands")
    parser.add_argument("--control_file", type=str, default=None,
                        help="Optional GUI-to-process control file for key commands")
    parser.add_argument("--num_steps", type=int, default=10000, help="Number of control steps")
    parser.add_argument("--terrain_difficulty", type=float, default=None,
                        help="Optional extra height scale (0-1) on top of the RTX artifact baseline.")
    parser.add_argument("--record", type=str, default=None, help="Record video to this path (EGL offscreen)")
    parser.add_argument("--terrain", type=str, default=None,
                        help="Artifact terrain selector: p3, p3_coarse, p3_fine. Legacy p3b aliases to p3_fine.")
    parser.add_argument("--phase", type=str, default=None,
                        help="Phase or sub-phase ID (p1/p1_fine/p2/p2_fine/p3/p3_fine). Explicit --terrain takes priority.")
    parser.add_argument("--flat", action="store_true",
                        help="Force flat ground (overrides terrain from --phase)")
    parser.add_argument("--random_spawn", action="store_true",
                        help="Randomize reset x/y/yaw. On terrain, z is auto-lifted from sampled ground height.")
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
        self._user32 = None
        self._async_keys = {}
        self._async_prev_down = {}
        self.control_file = None
        self._control_offset = 0
        if os.name == "nt":
            self._init_windows_async_keys()

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
        if self._poll_control_file():
            return
        if self.control_file is not None:
            return
        if self._poll_windows_async_keys():
            return
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

    def set_control_file(self, path):
        if not path:
            return
        self.control_file = Path(path)
        self.control_file.parent.mkdir(parents=True, exist_ok=True)
        self.control_file.touch(exist_ok=True)
        self._control_offset = self.control_file.stat().st_size

    def _poll_control_file(self):
        if self.control_file is None or not self.control_file.exists():
            return False
        try:
            current_size = self.control_file.stat().st_size
            if current_size < self._control_offset:
                self._control_offset = 0
            if current_size == self._control_offset:
                return False
            handled = False
            with self.control_file.open("r", encoding="utf-8") as handle:
                handle.seek(self._control_offset)
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    parts = line.split("\t", 1)
                    key = parts[1] if len(parts) == 2 else parts[0]
                    self._apply_key(key.strip())
                    handled = True
                self._control_offset = handle.tell()
            return handled
        except OSError:
            return False

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

    def _init_windows_async_keys(self):
        """Use Win32 async key polling so focused MuJoCo windows still receive control input."""
        try:
            self._user32 = ctypes.windll.user32
        except Exception:
            self._user32 = None
            return

        self._async_keys = {
            0x26: "up",         # VK_UP
            0x28: "down",       # VK_DOWN
            0x25: "left",       # VK_LEFT
            0x27: "right",      # VK_RIGHT
            0x51: "q",          # Q
            0x45: "e",          # E
            0x20: "space",      # VK_SPACE
            0x1B: "esc",        # VK_ESCAPE
        }
        if self.manual_mode:
            self._async_keys.update({
                0x57: "w",          # W
                0x53: "s",          # S
                0x52: "r",          # R
                0x09: "tab",        # VK_TAB
                0xDB: "lbracket",   # VK_OEM_4  [
                0xDD: "rbracket",   # VK_OEM_6  ]
            })

        self._async_prev_down = {vk: False for vk in self._async_keys}

    def _poll_windows_async_keys(self):
        if not self._user32:
            return False

        handled = False
        for vk, key_name in self._async_keys.items():
            is_down = bool(self._user32.GetAsyncKeyState(vk) & 0x8000)
            was_down = self._async_prev_down.get(vk, False)
            if is_down and not was_down:
                self._apply_key(key_name)
                handled = True
            self._async_prev_down[vk] = is_down
        return handled

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


def yaw_to_quat(yaw):
    half = 0.5 * yaw
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


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
    def __init__(self, mjcf_path, policy_runner=None, deploy_cfg=None, vel_cmd=None,
                 terrain_profile=None, terrain_difficulty=1.0, random_spawn=False):
        import mujoco

        self.policy = policy_runner
        self.manual_mode = policy_runner is None
        self.vel_cmd = np.array(vel_cmd) if vel_cmd is not None else np.array([0.5, 0.0, 0.0])
        self.random_spawn = random_spawn
        self.nominal_base_height = 0.69
        self.spawn_clearance_margin = 0.015
        self._rng = np.random.default_rng()
        self.terrain_meta = None

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

        # Load model from the resolved RTX artifact baseline.
        if terrain_profile and RTX_ARTIFACT_BASELINES[terrain_profile]["terrain_type"] is not None:
            self.model = load_model_with_terrain(mjcf_path, terrain_profile, difficulty=terrain_difficulty)
            profile = RTX_ARTIFACT_BASELINES[terrain_profile]
            nrow, ncol, half_x, half_y, max_elev, terrain_data = generate_terrain_data(
                terrain_profile, difficulty=terrain_difficulty
            )
            self.terrain_meta = {
                "artifact_key": terrain_profile,
                "nrow": nrow,
                "ncol": ncol,
                "half_x": half_x,
                "half_y": half_y,
                "max_elev": max_elev,
                "terrain_data": terrain_data,
                "layout": profile["layout"],
            }
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
        self.foot_geom_ids = []
        for foot_name in ("l_foot", "r_foot"):
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, foot_name)
            if geom_id >= 0:
                self.foot_geom_ids.append(geom_id)

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

    def _sample_spawn_pose(self):
        if self.terrain_meta is not None:
            return self._find_zero_height_spawn_pose(randomize=self.random_spawn)

        if self.random_spawn:
            x = self._rng.uniform(-0.5, 0.5)
            y = self._rng.uniform(-0.5, 0.5)
            yaw = self._rng.uniform(-math.pi, math.pi)
            return x, y, yaw
        return 0.0, 0.0, 0.0

    def _find_zero_height_spawn_pose(self, randomize):
        half_x = self.terrain_meta["half_x"]
        half_y = self.terrain_meta["half_y"]
        x_margin = min(1.0, half_x * 0.15)
        y_margin = min(0.5, half_y * 0.2)

        if randomize:
            for _ in range(400):
                x = self._rng.uniform(-half_x + x_margin, half_x - x_margin)
                y = self._rng.uniform(-half_y + y_margin, half_y - y_margin)
                yaw = self._rng.uniform(-math.pi, math.pi)
                if self._pose_supports_zero_height_ground(x, y, yaw):
                    return x, y, yaw
        else:
            for radius_scale in np.linspace(0.0, 1.0, 12):
                for yaw in (0.0, math.pi * 0.5, math.pi, -math.pi * 0.5):
                    x = (half_x - x_margin) * 0.1 * radius_scale
                    y = (half_y - y_margin) * 0.1 * radius_scale
                    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (0, 1), (-1, 0), (0, -1)):
                        px = x * sx
                        py = y * sy
                        if self._pose_supports_zero_height_ground(px, py, yaw):
                            return px, py, yaw

        print("[WARN] Could not find a strict zero-height spawn patch; falling back to origin with auto-lift")
        return 0.0, 0.0, 0.0

    def _set_pose_for_spawn_check(self, spawn_x, spawn_y, spawn_yaw):
        import mujoco

        self.data.qpos[0] = spawn_x
        self.data.qpos[1] = spawn_y
        self.data.qpos[2] = self.nominal_base_height
        self.data.qpos[3:7] = yaw_to_quat(spawn_yaw)
        for i, addr in enumerate(self.leg_qpos_addr):
            self.data.qpos[addr] = self.default_joint_pos[i]
        mujoco.mj_forward(self.model, self.data)

    def _pose_supports_zero_height_ground(self, spawn_x, spawn_y, spawn_yaw):
        self._set_pose_for_spawn_check(spawn_x, spawn_y, spawn_yaw)
        if not self.foot_geom_ids:
            return abs(self._terrain_height_at(spawn_x, spawn_y)) <= 1e-9
        for geom_id in self.foot_geom_ids:
            corners = self._foot_bottom_corners_world(geom_id)
            center = self.data.geom_xpos[geom_id].copy()
            samples = corners + [center]
            for sample in samples:
                if abs(self._terrain_height_at(sample[0], sample[1])) > 1e-9:
                    return False
        return True

    def _terrain_height_at(self, x, y):
        if self.terrain_meta is None:
            return 0.0
        half_x = self.terrain_meta["half_x"]
        half_y = self.terrain_meta["half_y"]
        nrow = self.terrain_meta["nrow"]
        ncol = self.terrain_meta["ncol"]
        terrain_data = self.terrain_meta["terrain_data"]
        max_elev = self.terrain_meta["max_elev"]

        x_norm = np.clip((x + half_x) / (2.0 * half_x), 0.0, 1.0)
        y_norm = np.clip((y + half_y) / (2.0 * half_y), 0.0, 1.0)
        col = min(ncol - 1, max(0, int(round(x_norm * (ncol - 1)))))
        row = min(nrow - 1, max(0, int(round(y_norm * (nrow - 1)))))
        return float(terrain_data[row, col] * max_elev)

    def _foot_bottom_corners_world(self, geom_id):
        size = self.model.geom_size[geom_id].copy()
        xpos = self.data.geom_xpos[geom_id].copy()
        xmat = self.data.geom_xmat[geom_id].reshape(3, 3)
        corners = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                local = np.array([sx * size[0], sy * size[1], -size[2]], dtype=np.float64)
                corners.append(xpos + xmat @ local)
        return corners

    def _compute_spawn_lift(self):
        if self.terrain_meta is None or not self.foot_geom_ids:
            return 0.0
        max_penetration = 0.0
        for geom_id in self.foot_geom_ids:
            for corner in self._foot_bottom_corners_world(geom_id):
                terrain_z = self._terrain_height_at(corner[0], corner[1])
                max_penetration = max(max_penetration, terrain_z - corner[2])
        return max(0.0, max_penetration + self.spawn_clearance_margin)

    def _apply_spawn_pose(self):
        import mujoco

        spawn_x, spawn_y, spawn_yaw = self._sample_spawn_pose()
        self.data.qpos[0] = spawn_x
        self.data.qpos[1] = spawn_y
        self.data.qpos[2] = self.nominal_base_height
        self.data.qpos[3:7] = yaw_to_quat(spawn_yaw)
        for i, addr in enumerate(self.leg_qpos_addr):
            self.data.qpos[addr] = self.default_joint_pos[i]
        mujoco.mj_forward(self.model, self.data)

        lift = self._compute_spawn_lift()
        if lift > 0.0:
            self.data.qpos[2] += lift
            mujoco.mj_forward(self.model, self.data)

        spawn_kind = "random" if self.random_spawn else "fixed"
        print(
            f"[RESET] {spawn_kind} spawn x={self.data.qpos[0]:+.2f} y={self.data.qpos[1]:+.2f} "
            f"yaw={spawn_yaw:+.2f} z={self.data.qpos[2]:.3f}"
            + (f" lift={lift:.3f}" if self.terrain_meta is not None else "")
        )

    def _warmup_stand(self, num_cycles, history_reset):
        import mujoco

        self.last_action = np.zeros(12)
        self.last_torques = np.zeros(12)
        self.sim_time = 0.0
        if history_reset:
            obs = self._build_obs_frame()
            self.obs_buffer.reset(obs)
        for _ in range(num_cycles):
            tgt = self.default_joint_pos
            for _ in range(DECIMATION):
                cp = np.array([self.data.qpos[a] for a in self.leg_qpos_addr])
                cv = np.array([self.data.qvel[a] for a in self.leg_dof_addr])
                torques = np.clip(self.kp * (tgt - cp) - self.kd * cv, -self.effort_limits, self.effort_limits)
                for i, act_id in enumerate(self.leg_actuator_ids):
                    self.data.ctrl[act_id] = torques[i]
                mujoco.mj_step(self.model, self.data)
            self.sim_time += CONTROL_DT
            if history_reset:
                obs = self._build_obs_frame()
                self.obs_buffer.append(obs)

    def reset(self):
        import mujoco
        self.vel_cmd[:] = self._initial_vel_cmd
        mujoco.mj_resetData(self.model, self.data)
        self._apply_spawn_pose()
        self._warmup_stand(num_cycles=HISTORY_LENGTH, history_reset=True)

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

    def reset_manual(self):
        """Reset robot pose for manual mode with randomized spawn and terrain-aware lift."""
        import mujoco

        mujoco.mj_resetData(self.model, self.data)
        self._apply_spawn_pose()
        self._warmup_stand(num_cycles=50, history_reset=False)

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
    kb_controller = None
    if args.keyboard or args.manual:
        kb_controller = KeyboardController(
            initial_vel=vel_cmd,
            manual_mode=args.manual,
            default_joint_pos=DEFAULT_JOINT_POS,
        )
        kb_controller.set_control_file(args.control_file)

    # RTX artifact baseline selection: --flat > --terrain > --phase
    terrain_profile = resolve_artifact_baseline(args.phase, args.terrain, args.flat)
    terrain_diff = args.terrain_difficulty if args.terrain_difficulty is not None else 1.0
    if terrain_profile is not None:
        profile = RTX_ARTIFACT_BASELINES[terrain_profile]
        if profile["terrain_type"] is None:
            print(f"[INFO] Using RTX artifact baseline '{terrain_profile}' -> flat ground")
        else:
            print(f"[INFO] Using RTX artifact baseline '{terrain_profile}' -> {profile['terrain_type']} terrain")
            if args.terrain_difficulty is None:
                print(f"[INFO] Base random_grid height max from RTX artifact: "
                      f"{profile['random_grid_height_range'][1]:.2f}m")
    elif args.flat:
        print("[INFO] --flat forced: using flat ground")

    env = MuJoCoDeploy(args.mjcf, policy, deploy_cfg=deploy_cfg, vel_cmd=vel_cmd,
                        terrain_profile=terrain_profile, terrain_difficulty=terrain_diff,
                        random_spawn=args.random_spawn)
    print(f"[INFO] MuJoCo model loaded from {args.mjcf}" +
          (f" with artifact_baseline={terrain_profile}" if terrain_profile else " (flat ground)"))
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
    print(f"[INFO] Running {args.num_steps} steps...")
    try:
        for step in range(args.num_steps):
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
