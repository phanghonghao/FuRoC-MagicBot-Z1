# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Custom play script for MagicBot Z1 12DOF locomotion with video recording (bypasses Hydra).

Supports two inference modes:
  --checkpoint model.pt  →  OnPolicyRunner (rsl-rl, existing behavior)
  --policy policy.pt     →  JIT inference (from spark_play.py, no rsl-rl dependency)

Camera tracking and renderer warmup are ported from spark_play.py (only active with --video).
"""

import argparse
import os
import sys
import time
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play MagicBot Z1 12DOF locomotion RL agent with video recording.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (OnPolicyRunner mode).")
parser.add_argument("--policy", type=str, default=None, help="Path to JIT-exported policy (.pt).")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--video", action="store_true", default=False, help="Record video.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video (in steps).")
parser.add_argument("--max_steps", type=int, default=800, help="Max steps when not recording video.")
parser.add_argument("--no_camera_track", action="store_true", default=False,
                    help="Disable camera tracking (camera stays at default position).")
parser.add_argument("--camera_distance", type=float, default=3.5,
                    help="Camera distance from robot for tracking (default: 3.5).")
parser.add_argument("--camera_height", type=float, default=1.5,
                    help="Camera height above robot for tracking (default: 1.5).")
parser.add_argument("--vel_x", type=float, default=None, help="Override forward velocity command (m/s).")
parser.add_argument("--vel_y", type=float, default=None, help="Override lateral velocity command (m/s).")
parser.add_argument("--vel_yaw", type=float, default=None, help="Override yaw angular velocity command (rad/s).")
parser.add_argument("--no_vel_limit", action="store_true", default=False,
                    help="Disable velocity limit clipping (set limit_ranges = ranges).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Validate: exactly one of --checkpoint or --policy must be provided
if not args_cli.checkpoint and not args_cli.policy:
    parser.error("Must provide --checkpoint or --policy.")
if args_cli.checkpoint and args_cli.policy:
    parser.error("--checkpoint and --policy are mutually exclusive.")

# Enable cameras for video recording
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# === Imports after Isaac Sim launch ===
import torch
import numpy as np
import gymnasium as gym
import importlib
import importlib.metadata as metadata

import isaaclab_tasks  # noqa: F401
import magiclab_rl_lab.tasks  # noqa: F401 - registers Magiclab-Z1-12dof-Velocity

# RobotPlayEnvCfg lives under robots/z1/12dof/ (digit-starting dir, can't use import statement)
# Resolve dynamically through gym registration
TASK_NAME = "Magiclab-Z1-12dof-Velocity"
_spec = gym.spec(TASK_NAME)
_entry = _spec.kwargs.get("play_env_cfg_entry_point", _spec.kwargs["env_cfg_entry_point"])
_mod_path, _cls_name = _entry.rsplit(":", 1)
RobotPlayEnvCfg = getattr(importlib.import_module(_mod_path), _cls_name)
from magiclab_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg

# JIT mode flag
USE_JIT = args_cli.policy is not None

if not USE_JIT:
    from rsl_rl.runners import OnPolicyRunner
    try:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    except ImportError:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        handle_deprecated_rsl_rl_cfg = None

    CKPT = args_cli.checkpoint
    print(f"[INFO] Mode: OnPolicyRunner (checkpoint)", flush=True)
    print(f"[INFO] Loading checkpoint: {CKPT}", flush=True)
else:
    CKPT = args_cli.policy
    print(f"[INFO] Mode: JIT policy (no rsl-rl dependency)", flush=True)
    print(f"[INFO] Loading policy: {CKPT}", flush=True)


def _update_camera(cam_ctx, env, cam_dist, cam_height):
    """Update camera to follow env-0 robot."""
    robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
    cam_ctx.set_camera_view(
        eye=[robot_pos[0] + 1.0, robot_pos[1] + cam_dist, robot_pos[2] + cam_height],
        target=[robot_pos[0] + 0.5, robot_pos[1], robot_pos[2] + 0.5],
    )


def _override_velocity_config(env_cfg):
    """Override velocity command ranges and limits from CLI args.

    The env __init__ does: self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
    So we must set limit_ranges to the desired values (they become the effective sampling range).
    """
    if not hasattr(env_cfg, "commands") or not hasattr(env_cfg.commands, "base_velocity"):
        print("[WARN] No base_velocity command found in env_cfg, skipping velocity override.", flush=True)
        return

    vel_cmd = env_cfg.commands.base_velocity
    has_vel_override = args_cli.vel_x is not None or args_cli.vel_y is not None or args_cli.vel_yaw is not None

    if args_cli.no_vel_limit and not has_vel_override:
        # Disable limit: set limit_ranges = ranges (no clipping, keep original sampling range)
        orig_ranges = vel_cmd.ranges
        print(f"[INFO] --no_vel_limit: disabling velocity limit clipping", flush=True)
        print(f"  Original ranges:     x={orig_ranges.lin_vel_x}, y={orig_ranges.lin_vel_y}, yaw={orig_ranges.ang_vel_z}", flush=True)
        print(f"  Original limit_rng:  x={vel_cmd.limit_ranges.lin_vel_x}, y={vel_cmd.limit_ranges.lin_vel_y}, yaw={vel_cmd.limit_ranges.ang_vel_z}", flush=True)
        # Set limit_ranges = ranges so __init__ copy has no effect
        vel_cmd.limit_ranges.lin_vel_x = list(orig_ranges.lin_vel_x)
        vel_cmd.limit_ranges.lin_vel_y = list(orig_ranges.lin_vel_y)
        vel_cmd.limit_ranges.ang_vel_z = list(orig_ranges.ang_vel_z)
        print(f"  New limit_ranges:    x={vel_cmd.limit_ranges.lin_vel_x}, y={vel_cmd.limit_ranges.lin_vel_y}, yaw={vel_cmd.limit_ranges.ang_vel_z}", flush=True)

    if has_vel_override:
        vel_x = args_cli.vel_x if args_cli.vel_x is not None else vel_cmd.ranges.lin_vel_x[0]
        vel_y = args_cli.vel_y if args_cli.vel_y is not None else vel_cmd.ranges.lin_vel_y[0]
        vel_yaw = args_cli.vel_yaw if args_cli.vel_yaw is not None else vel_cmd.ranges.ang_vel_z[0]
        # Set both ranges and limit_ranges to fixed values
        vel_cmd.ranges.lin_vel_x = [vel_x, vel_x]
        vel_cmd.ranges.lin_vel_y = [vel_y, vel_y]
        vel_cmd.ranges.ang_vel_z = [vel_yaw, vel_yaw]
        vel_cmd.limit_ranges.lin_vel_x = [vel_x, vel_x]
        vel_cmd.limit_ranges.lin_vel_y = [vel_y, vel_y]
        vel_cmd.limit_ranges.ang_vel_z = [vel_yaw, vel_yaw]
        print(f"[INFO] Velocity override: x={vel_x}, y={vel_y}, yaw={vel_yaw}", flush=True)


def main():
    try:
        # === Env config (play mode) ===
        env_cfg = RobotPlayEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.seed is not None:
            env_cfg.seed = args_cli.seed

        # === Override velocity command config ===
        _override_velocity_config(env_cfg)

        log_dir = os.path.dirname(CKPT) if not USE_JIT else os.path.dirname(os.path.dirname(CKPT))
        if hasattr(env_cfg, "log_dir"):
            env_cfg.log_dir = log_dir

        # === Create environment with render_mode for video ===
        print("[INFO] Creating environment...", flush=True)
        if args_cli.video:
            env = gym.make(
                TASK_NAME,
                cfg=env_cfg,
                render_mode="rgb_array",
            )
        else:
            env = gym.make(TASK_NAME, cfg=env_cfg)
        print(f"[INFO] Obs space: {env.observation_space}, Action space: {env.action_space}", flush=True)

        # === Camera tracking setup (from spark_play.py) ===
        cam_ctx = None
        cam_dist = args_cli.camera_distance
        cam_height = args_cli.camera_height

        if not args_cli.no_camera_track:
            try:
                from isaaclab.sim import SimulationContext
                cam_ctx = SimulationContext.instance()
                _update_camera(cam_ctx, env, cam_dist, cam_height)
                robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
                print(f"[INFO] Camera tracking enabled (dist={cam_dist}, height={cam_height}). "
                      f"Robot at ({robot_pos[0]:.1f}, {robot_pos[1]:.1f}, {robot_pos[2]:.1f})", flush=True)
            except Exception as e:
                print(f"[WARN] Camera tracking setup failed: {e}. Camera will stay at default position.", flush=True)
                cam_ctx = None
        else:
            # Static angled view
            try:
                from isaaclab.sim import SimulationContext
                cam_ctx_static = SimulationContext.instance()
                robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
                cam_ctx_static.set_camera_view(
                    eye=[robot_pos[0] + 12.0, robot_pos[1] + 12.0, robot_pos[2] + 15.0],
                    target=[robot_pos[0], robot_pos[1], robot_pos[2]],
                )
                print("[INFO] Camera tracking disabled. Static angled view set (45 deg overhead).", flush=True)
            except Exception as e:
                print(f"[INFO] Camera tracking disabled. Static view setup failed: {e}.", flush=True)

        # === Branch: JIT vs OnPolicyRunner ===
        if USE_JIT:
            _run_jit(env, cam_ctx, cam_dist, cam_height, log_dir)
        else:
            _run_onpolicy(env, cam_ctx, cam_dist, cam_height, log_dir)

    except Exception as e:
        print(f"[ERROR] Exception: {e}", flush=True)
        traceback.print_exc()
        raise


def _run_jit(env, cam_ctx, cam_dist, cam_height, log_dir):
    """JIT policy inference path (ported from spark_play.py)."""
    device = args_cli.device
    policy = torch.jit.load(CKPT, map_location=device)
    policy.eval()
    print(f"[INFO] JIT policy loaded on {device}", flush=True)

    # Reset environment
    obs, info = env.reset()

    # === Renderer warmup (from spark_play.py) ===
    if args_cli.video:
        warmup_steps = 20
        print(f"[INFO] Renderer warmup: running {warmup_steps} steps to prime the viewport...", flush=True)
        with torch.no_grad():
            for _ in range(warmup_steps):
                policy_obs = obs["policy"] if isinstance(obs, dict) else obs
                actions = policy(policy_obs)
                if cam_ctx is not None:
                    try:
                        _update_camera(cam_ctx, env, cam_dist, cam_height)
                    except Exception:
                        pass
                obs, _, _, _, _ = env.step(actions)
        print("[INFO] Warmup done. Resetting environment for recording...", flush=True)
        obs, info = env.reset()

    # === Wrap for video recording (after warmup, so no black frames) ===
    if args_cli.video:
        video_folder = os.path.join(log_dir, "videos", "play")
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording video to: {video_folder}", flush=True)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    def get_policy_obs(observation):
        if isinstance(observation, dict):
            return observation["policy"]
        return observation

    # === Inference loop ===
    max_steps = args_cli.video_length if args_cli.video else args_cli.max_steps
    timestep = 0
    start_time = time.time()

    print(f"[INFO] Running rollout ({max_steps} steps, JIT mode)...", flush=True)

    while simulation_app.is_running():
        with torch.no_grad():
            policy_obs = get_policy_obs(obs)
            actions = policy(policy_obs)

            # Update camera to follow robot (env 0)
            if cam_ctx is not None:
                try:
                    _update_camera(cam_ctx, env, cam_dist, cam_height)
                except Exception:
                    pass

            obs, reward, terminated, truncated, info = env.step(actions)

        timestep += 1
        if timestep >= max_steps:
            break

    elapsed = time.time() - start_time
    print(f"[INFO] Rollout done. Steps: {timestep}, Time: {elapsed:.1f}s", flush=True)
    if elapsed > 0:
        print(f"[INFO] FPS: {timestep / elapsed:.0f}", flush=True)

    env.close()
    print("[INFO] Environment closed. Video saved.", flush=True)


def _run_onpolicy(env, cam_ctx, cam_dist, cam_height, log_dir):
    """OnPolicyRunner inference path (existing behavior)."""
    agent_cfg = BasePPORunnerCfg()
    if handle_deprecated_rsl_rl_cfg is not None:
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    # === Renderer warmup ===
    if args_cli.video:
        warmup_steps = 20
        print(f"[INFO] Renderer warmup: running {warmup_steps} steps to prime the viewport...", flush=True)
        temp_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        temp_runner = OnPolicyRunner(
            temp_wrapped, agent_cfg.to_dict(),
            log_dir=None, device=agent_cfg.device,
        )
        temp_runner.load(CKPT)
        temp_policy = temp_runner.get_inference_policy(device=env.unwrapped.device)

        obs = temp_wrapped.get_observations()
        with torch.no_grad():
            for _ in range(warmup_steps):
                actions = temp_policy(obs)
                if cam_ctx is not None:
                    try:
                        _update_camera(cam_ctx, env, cam_dist, cam_height)
                    except Exception:
                        pass
                obs, _, _, _ = temp_wrapped.step(actions)
        print("[INFO] Warmup done.", flush=True)

    # === Wrap for video recording ===
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording video to: {video_kwargs['video_folder']}", flush=True)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # === Wrap for rsl-rl ===
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # === Create runner and load model ===
    runner = OnPolicyRunner(
        env_wrapped, agent_cfg.to_dict(),
        log_dir=None, device=agent_cfg.device,
    )
    runner.load(CKPT)
    print("[INFO] Model loaded successfully.", flush=True)

    # === Get inference policy ===
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print("[INFO] Inference policy ready.", flush=True)

    # === Run rollout ===
    obs = env_wrapped.get_observations()
    timestep = 0
    max_steps = args_cli.video_length if args_cli.video else args_cli.max_steps

    start_time = time.time()
    print(f"[INFO] Running rollout (max {max_steps} steps, OnPolicyRunner mode)...", flush=True)

    while simulation_app.is_running():
        with torch.no_grad():
            actions = policy(obs)

            # Update camera to follow robot (env 0)
            if cam_ctx is not None:
                try:
                    _update_camera(cam_ctx, env, cam_dist, cam_height)
                except Exception:
                    pass

            obs, rewards, dones, info = env_wrapped.step(actions)

        timestep += 1
        if timestep >= max_steps:
            break

    elapsed = time.time() - start_time
    print(f"[INFO] Rollout done. Steps: {timestep}, Time: {elapsed:.1f}s", flush=True)
    if elapsed > 0:
        print(f"[INFO] FPS: {timestep / elapsed:.0f}", flush=True)

    env.close()
    print("[INFO] Environment closed. Video saved.", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
