# MagicBot Z1 -- RL Locomotion Framework

> MagicBot Z1 12DOF bipedal robot reinforcement learning locomotion training framework.

---

## 1. Project Overview

```
magiclab_rl_lab/
├── source/magiclab_rl_lab/magiclab_rl_lab/   # Core source package
│   ├── tasks/locomotion/                     # Task definitions (env, MDP, Agent)
│   ├── assets/robots/                        # Robot Articulation config
│   ├── data/robots/                          # URDF / Mesh files
│   └── utils/                                # Utility functions
├── scripts/                                  # Train/eval/export scripts
│   ├── rsl_rl/                               # RSL-RL training & play scripts
│   └── automation/                           # Automated multi-stage training orchestration
├── sim2sim/                                  # MuJoCo Sim2Sim verification
├── deploy/                                   # Robot deployment
├── training_plans/                           # Multi-stage training plan YAML
├── train_bash.sh                             # Training launch script
├── play_bash.sh                              # Evaluation launch script
└── logs/                                     # Training logs & checkpoints
```

### Two-Platform Architecture

```
+------------------------------------------------------------------+
|                  RTX 6000D (Remote Training Server)               |
|                   phh@192.168.120.155                             |
|           85GB VRAM | 16384 envs | Isaac Lab 0.47.2 + rsl_rl     |
|                                                                   |
|  [Training Engine]     [Model Management]     [Automation]        |
|   Isaac Lab             checkpoint             orchestrator       |
|   rsl_rl PPO            best_model             train_monitor      |
|                                                                   |
|  logs/rsl_rl/magiclab_z1_12dof_velocity/{run_dir}/               |
+------------------------------------------------------------------+
            |                              |
            | JIT/ONNX export              | model_*.pt + deploy.yaml
            | .mp4 videos                  |
            v                              v
+------------------------------------------------------------------+
|               Local Windows (Analysis & Organization)             |
|         D:\Desktop_Files\GPU-Train\RTX6000\Magicbot_Z1\           |
|                                                                   |
|  [videos/]        [docs/]              [/gpu-train CLI]           |
|   Version video    Documentation        Management commands       |
|   archive          analysis                                       |
+------------------------------------------------------------------+
```

### Platform Directory Details

#### Remote RTX Server (`phh@192.168.120.155`)

```
~/
├── magiclab_rl_lab/                    # Main code repository
│   ├── source/magiclab_rl_lab/magiclab_rl_lab/
│   │   ├── tasks/locomotion/
│   │   │   ├── robots/z1/12dof/
│   │   │   │   ├── velocity_env_cfg.py
│   │   │   │   └── velocity_env_cfg_s7_rough_full.py
│   │   │   ├── agents/rsl_rl_ppo_cfg.py
│   │   │   └── mdp/ (rewards, observations, curriculums, commands)
│   │   ├── utils/ (parser_cfg, export_deploy_cfg)
│   │   └── assets/robots/magiclab.py
│   ├── scripts/
│   │   ├── rsl_rl/ (train, train_multigpu, play, play_z1_video, play_keyboard)
│   │   ├── export_jit.py, train_monitor.py, plot_learning_curves.py
│   │   └── automation/ (orchestrator, training_launcher, stage_manager, config_swapper)
│   ├── sim2sim/ (mujoco_deploy, mujoco_sim2sim)
│   ├── deploy/robot_deploy.py
│   ├── training_plans/ (z1_5stage_plan, z1_s4_s5_plan)
│   └── logs/rsl_rl/magiclab_z1_12dof_velocity/
│       ├── best_models.json
│       └── <RUN_DIR>/model_*.pt, params/, exported/policy.pt
├── IsaacLab/                           # Isaac Lab framework (v0.47.2)
├── magicbot-z1_description/            # URDF + MJCF + meshes
├── mujoco_record_video.py             # MuJoCo EGL offscreen recording
└── miniconda3/envs/isaaclab/          # Conda environment
```

#### Local Windows (`D:\Desktop_Files\GPU-Train\RTX6000\`)

```
RTX6000/
├── Magicbot_Z1/
│   ├── magiclab_rl_lab/               # Code mirror
│   ├── magicbot-z1_description/       # Robot description mirror
│   ├── magicbot-z1_sdk/               # SDK (ARM64 + x86_64)
│   ├── configs/                       # Environment config backups
│   ├── scripts/ (compare_videos, label_video)
│   ├── docs/                          # Documentation
│   ├── videos/                        # Downloaded recordings
│   ├── best_models.json              # scp'd best model summary
│   └── IsaacLab/
├── launch_dual_training.sh
└── videos/
```

> Note: Training checkpoints (`model_*.pt`) exist only on the remote RTX server. Local machine holds code mirrors and downloaded result files.

---

## 2. Source Architecture

### 2.1 Module Dependencies

```
tasks/locomotion/
├── __init__.py                          # imports robots
├── agents/
│   ├── __init__.py
│   └── rsl_rl_ppo_cfg.py               # PPO hyperparameter config
├── mdp/
│   ├── __init__.py                      # imports Isaac Lab mdp
│   ├── observations.py                  # Custom observations (gait_phase, contact_mask)
│   ├── rewards.py                       # Custom reward functions (energy, feet_gait, etc.)
│   ├── curriculums.py                   # Curriculum learning (velocity command progression)
│   └── commands/
│       └── velocity_command.py          # Velocity command with limit_ranges
└── robots/
    ├── __init__.py                      # imports z1
    └── z1/
        ├── __init__.py                  # imports 12dof
        └── 12dof/
            ├── __init__.py              # Gym task registration
            └── velocity_env_cfg.py      # Complete environment config
```

### 2.2 Task Registration

`robots/z1/12dof/__init__.py` registers the task via Gymnasium:

```python
gym.register(
    id="Magiclab-Z1-12dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "...velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": "...velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "...rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
```

---

## 3. Environment Configuration

### 3.1 Scene (RobotSceneCfg)

| Component | Config |
|-----------|--------|
| Terrain | `TerrainImporterCfg`, generator mode, 50% flat |
| Terrain size | 8m x 8m, 9 rows x 21 columns |
| Ground friction | static=1.0, dynamic=1.0 |
| Robot | `MAGICLAB_Z1_12DOF_CFG`, URDF: `MagicBotZ1_12dof_arm_ready_pos.urdf` |
| Initial height | 0.69m |
| Contact sensor | Full body `.*`, history_length=3, track_air_time=True |
| Height scan | RayCaster on pelvis, Grid 1.6x1.0m, resolution 0.1m |
| Parallel envs | Default 16384, spacing 2.5m |

### 3.2 Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Physics dt | 0.002s (500Hz) |
| Decimation | 10 -> control freq 50Hz |
| Episode length | 20s (1000 steps) |
| Render interval | = decimation |

### 3.3 Observation Space (235-dim)

**Policy observation = 47-dim/frame x 5 frames history_length**

Isaac Lab uses **per-term interleaved** layout (not frame-stacked):
```
[ang_vel x5, gravity x5, cmd x5, jpos x5, jvel x5, act x5, gait x5]
```

| Observation | Dim | Scale | Noise |
|-------------|-----|-------|-------|
| `base_ang_vel` | 3 | x0.2 | U(-0.2, 0.2) |
| `projected_gravity` | 3 | -- | U(-0.1, 0.1) |
| `velocity_commands` | 3 | -- | -- |
| `joint_pos_rel` | 12 | -- | U(-0.02, 0.02) |
| `joint_vel_rel` | 12 | x0.05 | U(-1.5, 1.5) |
| `last_action` | 12 | x1.0, clip(-100, 100) | -- |
| `gait_phase` | 2 | -- | -- |

**Critic extra observations** (same 5-frame history):
- `base_lin_vel` (3) -- true linear velocity (privileged info)
- `contact_mask` (2) -- ankle contact force > 5N binary mask

> Note: `last_action` clip=(-100, 100) is effectively unclipped. MuJoCo and deployment scripts must match.

### 3.4 Gait Phase Calculation

```python
# Sinusoidal gait, period 0.6s
global_phase = (episode_length_buf * step_dt) % period / period
sin_pos = sin(2*pi * phase)

stance_mask[:, 0] = (sin_pos >= 0)   # Left foot stance
stance_mask[:, 1] = (sin_pos < 0)    # Right foot stance

# Standing still (cmd < 0.02): both feet stance
stance_mask[cmd_norm < 0.02] = [1, 1]
```

### 3.5 Action Space (12-dim)

```python
JointPositionActionCfg(
    joint_names=[left_hip_6joints, right_hip_6joints],
    scale=0.25,
    use_default_offset=True,   # offset = default_joint_pos
    preserve_order=True
)
```

**Joint order**:
```
[L_hip_pitch, L_hip_roll, L_hip_yaw, L_knee, L_ankle_pitch, L_ankle_roll,
 R_hip_pitch, R_hip_roll, R_hip_yaw, R_knee, R_ankle_pitch, R_ankle_roll]
```

### 3.6 Reward Functions (20+ terms)

#### Task Rewards

| Name | Weight | Function | Description |
|------|--------|----------|-------------|
| `track_lin_vel_xy` | +1.0 | `track_lin_vel_xy_yaw_frame_exp` | Linear velocity tracking (exp kernel, std=sqrt(0.25)) |
| `track_ang_vel_z` | +0.5 | `track_ang_vel_z_exp` | Angular velocity tracking (exp kernel) |
| `alive` | +0.15 | `is_alive` | Survival reward |

#### Penalties

| Name | Weight | Description |
|------|--------|-------------|
| `base_linear_velocity` | -2.0 | Z-axis linear velocity L2 |
| `base_angular_velocity` | -0.05 | Roll/Pitch angular velocity L2 |
| `joint_vel` | -0.001 | Joint velocity L2 |
| `joint_acc` | -2.5e-7 | Joint acceleration L2 |
| `action_rate` | -0.05 | Action smoothness L1 |
| `dof_pos_limits` | -5.0 | Joint limit penalty |
| `energy` | -2e-5 | |qvel x qfrc| energy cost |
| `joint_deviation_legs` | -0.7 | Hip roll/yaw deviation from default L1 |

#### Posture/Height

| Name | Weight | Description |
|------|--------|-------------|
| `flat_orientation_l2` | -5.0 | Upright torso penalty |
| `base_height` | -10.0 | Target height 0.7m L2 |
| `stand_still` | -3.5 | Joint deviation from default at low command (cmd < 0.05) |

#### Foot Rewards

| Name | Weight | Description |
|------|--------|-------------|
| `feet_contact_number` | +0.5 | Gait contact matching (consistent with gait_phase) |
| `feet_slide` | -0.2 | Foot sliding penalty |
| `feet_clearance` | +1.0 | Swing foot lift height reward (exp kernel, target=0.1m) |
| `undesired_contacts` | -1.0 | Non-ankle contact (force > 1N) |

### 3.7 Termination Conditions

| Condition | Description |
|-----------|-------------|
| `time_out` | Reached 20s episode length |
| `base_height` | Torso height < 0.2m |
| `bad_orientation` | Yaw angle deviation > 0.8 rad |

### 3.8 Curriculum Learning

| Name | Logic |
|------|-------|
| `terrain_levels` | Terrain difficulty progresses with training |
| `lin_vel_cmd_levels` | When track_lin_vel_xy reward > weight x 0.8, expand velocity range +/-0.1 |
| `ang_vel_cmd_levels` | When track_ang_vel_z reward > weight x 0.8, expand angular velocity range +/-0.1 |

### 3.9 Randomization (EventCfg)

| Event | Mode | Description |
|-------|------|-------------|
| `physics_material` | startup | Friction randomization (0.3, 1.0) |
| `add_base_mass` | startup | Pelvis mass scale x0.7~1.3 |
| `randomize_rigid_body_mass_others` | startup | Full body mass scale x0.7~1.3 |
| `base_external_force_torque` | reset | External force/torque (currently 0) |
| `reset_base` | reset | Randomize initial position/velocity |
| `reset_robot_joints` | reset | Randomize joint position/velocity |
| `push_robot` | interval(5s) | Push disturbance vel +/-0.5 m/s |

---

## 4. PPO Agent Configuration

**File**: `tasks/locomotion/agents/rsl_rl_ppo_cfg.py`

### Network Architecture

```
Actor:  Linear(235, 512) -> ELU -> Linear(512, 256) -> ELU -> Linear(256, 128) -> ELU -> Linear(128, 12)
Critic: Same structure (input 237-dim with contact_mask)
```

- Activation: ELU
- Initial noise std: 1.0

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| `num_steps_per_env` | 24 |
| `max_iterations` | 50000 |
| `save_interval` | 100 |
| `learning_rate` | 1e-3 (adaptive schedule) |
| `gamma` | 0.99 |
| `lam` (GAE lambda) | 0.95 |
| `clip_param` | 0.2 |
| `entropy_coef` | 0.01 |
| `value_loss_coef` | 1.0 |
| `num_learning_epochs` | 5 |
| `num_mini_batches` | 4 |
| `desired_kl` | 0.01 |
| `max_grad_norm` | 1.0 |
| `empirical_normalization` | False |
| `use_clipped_value_loss` | True |

### Batch Size Calculation

```
batch_size = num_envs x num_steps_per_env = 16384 x 24 = 393,216
mini_batch_size = batch_size / num_mini_batches = 98,304
```

---

## 5. Robot Parameters

### 5.1 URDF

**File**: `data/robots/magicbot-Z1/urdf/MagicBotZ1_12dof_arm_ready_pos.urdf`
(arm_ready_pos version, arms fixed at ready pose)

### 5.2 Joint Parameters

| Joint | KP | KD | Armature | Effort Limit |
|-------|------|------|----------|-------------|
| L/R hip_pitch | 100 | 4 | 0.02863 | 120 |
| L/R hip_roll | 100 | 4 | 0.02863 | 120 |
| L/R hip_yaw | 100 | 4 | 0.02863 | 120 |
| L/R knee | 150 | 5 | 0.02863 | 120 |
| L/R ankle_pitch | 60 | 3 | 0.01503 | 50 |
| L/R ankle_roll | 60 | 3 | 0.01503 | 50 |

### 5.3 Default Joint Positions (Standing Pose)

```python
default_pos = [-0.35, 0, 0, 0.7, -0.35, 0] x 2  # Left and right symmetric
#              pitch  roll yaw knee  pitch  roll
```

### 5.4 Action Scaling

- Action scale = 0.25
- `target_pos = default_pos + action x 0.25`
- Joint velocity limits: legs 20 rad/s, ankles 15 rad/s

---

## 6. Training Scripts

### 6.1 train.py

**File**: `scripts/rsl_rl/train.py`

```
python scripts/rsl_rl/train.py \
    --task Magiclab-Z1-12dof-Velocity \
    --headless \
    --num_envs 16384 \
    --device cuda:0 \
    --run_name z1_locomotion_v1 \
    --max_iterations 50000
```

**Resume training**:
```
python scripts/rsl_rl/train.py \
    --task Magiclab-Z1-12dof-Velocity \
    --headless \
    --resume \
    --load_run <timestamp>_<run_name> \
    --checkpoint model_<N>.pt
```

**Flow**:
1. AppLauncher starts Isaac Sim
2. `gym.make("Magiclab-Z1-12dof-Velocity")` creates environment
3. `RslRlVecEnvWrapper` wraps it
4. `OnPolicyRunner` creates PPO trainer
5. Before training, auto-calls `export_deploy_cfg()` to generate `deploy.yaml`
6. `runner.learn()` starts training

**Launch script**: `train_bash.sh` (nohup background, log to `train_z1.log`)

### 6.2 play.py

**File**: `scripts/rsl_rl/play.py`

```
python scripts/rsl_rl/play.py \
    --task Magiclab-Z1-12dof-Velocity \
    --num_envs 32 \
    --checkpoint model_<N>.pt
```

- Uses `OnPolicyRunner` to load checkpoint
- Auto-exports JIT/ONNX model
- Supports video recording with `--video`

### 6.3 play_keyboard.py

**File**: `scripts/rsl_rl/play_keyboard.py`

- WASD/QE keyboard velocity command control
- Camera follow mode

### 6.4 play_z1_video.py

**File**: `scripts/rsl_rl/play_z1_video.py`

- Bypasses Hydra, loads env config directly
- Optimized for video recording
- `--video --video_length 200`

---

## 7. Model Export

### 7.1 export_jit.py

**File**: `scripts/export_jit.py`

```
python scripts/export_jit.py \
    --checkpoint logs/.../model_5000.pt \
    --onnx
```

**Features**:
- Extracts actor weights from checkpoint
- Supports rsl-rl 3.x (`model_state_dict`) and 5.x (`actor_state_dict`)
- Auto-infers network structure [235, 512, 256, 128, 12]
- Exports JIT (`policy.pt`) and optional ONNX (`policy.onnx`)
- Pure PyTorch, no Isaac Sim required

**Output path**: `{checkpoint_dir}/exported/policy.pt`

### 7.2 deploy.yaml Auto-generation

**File**: `source/.../utils/export_deploy_cfg.py`

Auto-called at training start, generates `params/deploy.yaml` containing:

| Field | Content |
|-------|---------|
| `joint_ids_map` | SDK joint name to training joint index mapping |
| `step_dt` | Control step (0.02s) |
| `stiffness` / `damping` | PD gains |
| `default_joint_pos` | Default joint positions |
| `actions` | Action scale, offset, clip |
| `observations` | Observation dimensions, scale, clip |
| `commands` | Velocity command ranges |

---

## 8. Training Monitoring

### 8.1 train_monitor.py

**File**: `scripts/train_monitor.py`

```
# Continuous monitoring (run alongside training)
python scripts/train_monitor.py \
    --log_root logs/rsl_rl/magiclab_z1_12dof_velocity \
    --terrain gentle --poll_interval 120

# One-shot analysis of completed run
python scripts/train_monitor.py --once \
    --run_dir logs/.../<run_dir>

# Auto-export best model
python scripts/train_monitor.py --auto_export --terrain gentle
```

**5 overfitting checks**:
1. **Reward decline**: sustained decay after best reward
2. **action_rate anomaly**: exceeds terrain threshold
3. **Std collapse**: policy standard deviation too low
4. **Value loss anomaly**: value function loss spike
5. **Entropy anomaly**: policy entropy too low

**Terrain threshold config**:
| Terrain | Expected Best Range | Reward Ceiling | action_rate Warning |
|---------|-------------------|---------------|-------------------|
| flat | 15k~30k | 49.0 | -0.8 |
| gentle | 20k~35k | 47.0 | -1.0 |
| rough | 25k~40k | 38.0 | -1.5 |

### 8.2 Automation Orchestrator

**File**: `scripts/automation/`

```
python -m automation.orchestrator \
    --plan training_plans/z1_5stage_plan.yaml \
    --start-from s4_rough_l1
```

**Modules**:
| File | Responsibility |
|------|---------------|
| `orchestrator.py` | Main event loop, manages stage transitions |
| `stage_manager.py` | YAML training plan parsing |
| `config_swapper.py` | Environment config switching between stages |
| `training_launcher.py` | Subprocess training launch |
| `embedded_monitor.py` | Embedded overfitting monitoring |
| `state_store.py` | Crash recovery state persistence |

**Features**:
- Auto-resume from previous stage best model
- Overfitting detection auto-stops current stage
- Crash recovery (state persisted to JSON)
- Retry: NaN reduces learning rate x0.5, OOM halves env count

---

## 9. Multi-Stage Training Plan

**File**: `training_plans/z1_5stage_plan.yaml`

```
S1_flat (10k) -> S2_flat (25k) -> S3_gentle (25k) -> S4_rough_l1 (40k) -> S4_full_terrain (50k) -> S5_full (50k)
```

| Stage | Terrain | Iterations | Resume From |
|-------|---------|-----------|-------------|
| `s1_flat` | flat | 10,000 | -- |
| `s2_flat` | flat | 25,000 | s1_flat |
| `s3_gentle` | gentle | 25,000 | s2_flat |
| `s4_rough_l1` | rough | 40,000 | s3_gentle |
| `s4_full_terrain` | rough | 50,000 | s4_rough_l1 |
| `s5_full` | rough | 50,000 | s4_full_terrain |

Each stage adjusts terrain config and reward weights via independent `configs/velocity_env_cfg_s{N}_*.py`.

---

## 10. Sim2Sim Verification

### 10.1 mujoco_deploy.py

**File**: `sim2sim/mujoco_deploy.py`

```
python sim2sim/mujoco_deploy.py \
    --mjcf ../magicbot-z1_description/mjcf/MAGICBOTZ1.xml \
    --policy logs/.../exported/policy.pt \
    --deploy_cfg logs/.../params/deploy.yaml \
    --keyboard
```

**Features**:
- Loads JIT/ONNX policy + deploy.yaml
- Isaac Lab per-term observation history layout (`ObservationBuffer`)
- Matches training PD gains
- Keyboard velocity control
- Fall detection and auto-reset

### 10.2 mujoco_sim2sim.py

**File**: `sim2sim/mujoco_sim2sim.py`

- Class-based architecture (`Z1SimConfig`, `Z1RobotConfig`, `Z1ObsConfig`)
- EGL offscreen video recording (no X Server required)
- Clean `run_mujoco(policy, cfg)` entry point

### 10.3 Key Implementation Details

MuJoCo and Isaac Lab consistency requirements:
1. **Observation history layout**: per-term interleaved (not frame-stacked)
2. **PD gains**: read from deploy.yaml, match training values
3. **Armature**: consistent with Isaac Lab
4. **Joint damping**: clear default damping in MuJoCo model (set to 0)
5. **Gait phase**: match training sin-based gait calculation
6. **Control frequency**: physics 500Hz, policy 50Hz (decimation=10)

---

## 11. Robot Deployment

**File**: `deploy/robot_deploy.py`

```
python deploy/robot_deploy.py \
    --policy policy.onnx \
    --deploy_cfg deploy.yaml \
    --robot_ip 192.168.54.111 \
    --suspension_test     # Suspension test mode
```

**Safety protocol**:
1. Suspension test mode to verify policy output
2. Initial velocity command is zero
3. Gradually increase velocity commands
4. Ctrl+C emergency stop

**Architecture**:
- Thread-safe sensor data (IMU, joint states)
- Policy inference at 50Hz / joint commands at 500Hz
- Loads ONNX or JIT models

---

## 12. Training History

### 12.1 Training Chain

```
s1_flat (m3861, r=47.33, flat)
  └──> s2_gentle (m47862, r=47.06, gentle) <-- Historical best, HEALTHY
         └──> s3_rough_l2 (m32790, r=38.04, rough) <-- Best on rough terrain
                └──> s4_full_terrain (m5155, r=37.73, full terrain) <-- Training
```

### 12.2 Retained Training Runs (4)

| Run Directory | Best Model | Best Reward | Status |
|---------------|-----------|-------------|--------|
| `2026-04-30_04-53-17_z1_locomotion_s1_flat` | model_3861 | 47.33 | OVERFITTING (resume source) |
| `2026-05-01_04-50-05_z1_locomotion_s4_gentle_terrain` | model_47862 | **47.06** | HEALTHY |
| `2026-05-01_07-04-35_z1_locomotion_s5_rough_terrain` | model_32790 | 38.04 | OVERFITTING |
| `2026-05-04_16-56-05_s4_full_terrain` | model_5155 | 37.73 | RUNNING |

---

## 13. Complete File Index

### Core Source

| File | Path | Description |
|------|------|-------------|
| Env config | `source/.../tasks/locomotion/robots/z1/12dof/velocity_env_cfg.py` | Scene, full MDP config |
| Agent config | `source/.../tasks/locomotion/agents/rsl_rl_ppo_cfg.py` | PPO hyperparameters |
| Observations | `source/.../tasks/locomotion/mdp/observations.py` | gait_phase, contact_mask |
| Rewards | `source/.../tasks/locomotion/mdp/rewards.py` | Custom reward functions |
| Curriculum | `source/.../tasks/locomotion/mdp/curriculums.py` | Velocity progression |
| Velocity command | `source/.../tasks/locomotion/mdp/commands/velocity_command.py` | With limit_ranges |
| Robot config | `source/.../assets/robots/magiclab.py` | URDF, PD gains, joint mapping |
| Export utility | `source/.../utils/export_deploy_cfg.py` | Auto-generate deploy.yaml |
| Task registration | `source/.../tasks/locomotion/robots/z1/12dof/__init__.py` | Gym registration |

### Scripts

| File | Path | Description |
|------|------|-------------|
| Training | `scripts/rsl_rl/train.py` | RSL-RL training entry |
| Evaluation | `scripts/rsl_rl/play.py` | OnPolicyRunner evaluation |
| Keyboard eval | `scripts/rsl_rl/play_keyboard.py` | Keyboard control |
| Video recording | `scripts/rsl_rl/play_z1_video.py` | Hydra-bypass video recording |
| JIT export | `scripts/export_jit.py` | Checkpoint to JIT/ONNX |
| Train monitor | `scripts/train_monitor.py` | Overfitting detection |
| Orchestrator | `scripts/automation/orchestrator.py` | Multi-stage training orchestration |

### Deployment & Sim2Sim

| File | Path | Description |
|------|------|-------------|
| MuJoCo deploy | `sim2sim/mujoco_deploy.py` | Sim2Sim verification |
| MuJoCo recording | `sim2sim/mujoco_sim2sim.py` | EGL video recording |
| Robot deploy | `deploy/robot_deploy.py` | Physical robot control |

### Shell Scripts

| File | Description |
|------|-------------|
| `train_bash.sh` | nohup background training launch |
| `play_bash.sh` | Evaluation launch |
| `play_keyboard_bash.sh` | Keyboard control evaluation |
| `magiclab_rl_lab.sh` | Isaac Lab environment launch |

---

## 14. Remote Connection

- **RTX 6000D**: `phh@192.168.120.155`, VPN: iNode
- See: [Remote_GPU_Server_Connection.md](../../docs/Remote_GPU_Server_Connection.md)

---

## 15. Common Commands

### RTX Server

```bash
# Training status
ssh phh@192.168.120.155 'ps aux | grep train | grep phh | grep -v grep'

# GPU usage
ssh phh@192.168.120.155 'nvidia-smi'

# Training log
ssh phh@192.168.120.155 'tail -30 /tmp/z1_train_s6.log'

# List models
ssh phh@192.168.120.155 'ls -lh ~/magiclab_rl_lab/logs/rsl_rl/magiclab_z1_12dof_velocity/<RUN>/model_*.pt'

# Export JIT
ssh phh@192.168.120.155 "cd ~/magiclab_rl_lab && source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab && python scripts/export_jit.py --checkpoint logs/.../model_N.pt"

# Overfitting detection
ssh phh@192.168.120.155 "cd ~/magiclab_rl_lab && source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab && python scripts/train_monitor.py --once --terrain gentle"

# EGL pre-check (Isaac Sim requires NVIDIA EGL vendor file)
ssh phh@192.168.120.155 'test -f ~/miniconda3/envs/isaaclab/share/glvnd/egl_vendor.d/10_nvidia.json && echo "OK" || echo "MISSING"'
```

### Local

```bash
# Video label
python RTX6000/Magicbot_Z1/scripts/label_video.py video.mp4 --model model_N --run s4_gentle --reward 47.06

# 2x2 comparison
python RTX6000/Magicbot_Z1/scripts/compare_videos.py v1.mp4 v2.mp4 v3.mp4 v4.mp4 --labels "A" "B" "C" "D"
```

### Key Config File Quick Reference

| File | Path | Controls |
|------|------|----------|
| Env config | `source/.../z1/12dof/velocity_env_cfg.py` | Reward weights, observations, terrain, PD params, action scale |
| PPO config | `source/.../agents/rsl_rl_ppo_cfg.py` | Learning rate, entropy coef, GAE, network structure |
| Reward functions | `source/.../mdp/rewards.py` | Mathematical definitions of each reward term |
| Observation defs | `source/.../mdp/observations.py` | Meaning of each observation vector dimension |
| Training plan | `training_plans/z1_5stage_plan.yaml` | Stage order, iterations, config switching |
| Rough terrain config | `source/.../z1/12dof/velocity_env_cfg_s7_rough_full.py` | v7 rough terrain specific config |

---

## Appendix A: Checkpoint vs JIT

Both files use `.pt` extension but have entirely different content and purpose.

### Comparison

| | `model_NNNNN.pt` (Checkpoint) | `policy.pt` (JIT Export) |
|---|---|---|
| **Size** | ~6.8 MB | ~1.2 MB |
| **Content** | Actor + Critic + optimizer state | Actor only (inference) |
| **Dependency** | Requires `rsl-rl` library | Pure `torch.jit.load()`, no dependency |
| **Usage** | Resume training (`--load_run`), analyze training state | Deploy to robot, record video, sim2sim |
| **Generated by** | Training script every 100 iter | `export_jit.py` extracts Actor from checkpoint |
| **Can resume training?** | Yes | No |
| **rsl-rl version** | 3.x and 5.x incompatible | Universal across versions |
| **Load method** | `OnPolicyRunner.load()` | `torch.jit.load()` |

### Why JIT Exists

rsl-rl 3.x saves ActorCritic as `model_state_dict`, while 5.x separates into `actor_state_dict` + `critic_state_dict` -- these are incompatible. JIT export produces a pure `torch.jit.script` model that any platform and any rsl-rl version can load, bypassing version issues entirely.

### Export Command

```bash
python scripts/export_jit.py --checkpoint logs/rsl_rl/<RUN>/model_5000.pt
# Output: logs/rsl_rl/<RUN>/exported/policy.pt (~1.2 MB)
```

The script auto-infers network structure (obs_dim, hidden_dims, action_dim) from checkpoint state_dict.

### Disk Space

Training saves a checkpoint every 100 iterations; a full 50K-iteration run produces ~500 files (~3.4 GB). After JIT export (~1.2 MB), original checkpoints can be safely deleted to save space (but deletion prevents resume training).

> Current status: s1_flat and s3_rough_l2 original checkpoints have been deleted, only JIT policy remains. s2_gentle and s4_full_terrain have both.

---

## Appendix B: Training Metrics Reference

Training metrics fall into two categories: cross-run comparable (behavior metrics) and same-run only (training metrics). Evaluate policy quality with behavior metrics; assess training trends with training metrics.

### Cross-Run Comparable Metrics (Behavior Metrics)

These are independent of reward weight settings and directly comparable across runs and terrain configs. Use for evaluating policy quality and selecting best checkpoints.

| Metric | TensorBoard Tag | Range | Direction | Ideal | Description |
|--------|----------------|-------|-----------|-------|-------------|
| **time_out** | `Episode_Termination/time_out` | 0~1 | Higher better | >0.8 | Episode completes full 20s |
| **ep_len** | `Train/mean_episode_length` | 0~1000 | Higher better | >950 | Mean episode steps, 1000 = full |
| **bad_ori** | `Episode_Termination/bad_orientation` | 0~1 | Lower better | <0.1 | Robot falls (abnormal orientation) |
| **vel_err** | `Metrics/base_velocity/error_vel_xy` | 0+ m/s | Lower better | <0.3 | Linear velocity tracking error |
| **vel_yaw_err** | `Metrics/base_velocity/error_vel_yaw` | 0+ rad/s | Lower better | <0.5 | Yaw angular velocity tracking error |

### Quick Evaluation Standard

| Grade | time_out | ep_len | bad_ori | vel_err |
|-------|----------|--------|---------|---------|
| Excellent | >90% | >950 | <5% | <0.3 |
| Good | >70% | >800 | <15% | <0.5 |
| Average | >40% | >500 | <30% | <0.8 |
| Poor | <40% | <500 | >30% | >0.8 |

### Same-Run Only Metrics (Training Metrics)

These depend on specific reward weights and environment config. Only meaningful for trend analysis within the same run.

| Metric | TensorBoard Tag | Use | Description |
|--------|----------------|-----|-------------|
| **mean_reward** | `Train/mean_reward` | Training trend | Not comparable across runs (different weights). Within run: up = improving, down = possible overfitting |
| **action_rate** | `Episode_Reward/action_rate` | Detect policy collapse | < -1.0 usually means severe jitter, possible collapse |
| **entropy** | `Loss/entropy` | Monitor exploration | High to low is normal. Prematurely reaching 0 = insufficient exploration |
| **policy_std** | `Train/mean_std` | Action determinism | < 0.01 means near-deterministic output, possible overfitting |
| **value_loss** | `Loss/value_function` | Value function quality | Sudden spike = value function divergence |
| **surrogate_loss** | `Loss/surrogate` | Policy update magnitude | Should be near 0; large deviation = unstable update |
| **terrain_levels** | `Curriculum/terrain_levels` | Curriculum progress | Current terrain difficulty level |
| **vel_cmd_levels** | `Curriculum/lin_vel_cmd_levels` | Curriculum progress | Velocity command difficulty level |
| **throughput** | `Perf/total_fps` | Training efficiency | steps/s, for ETA estimation |
| **collection_time** | `Perf/collection_time` | Efficiency diagnosis | Simulation data collection time |
| **learning_time** | `Perf/learning_time` | Efficiency diagnosis | PPO policy update time |

### Trend Judgment

| Trend | Signal | Meaning |
|-------|--------|---------|
| Reward up + time_out up + bad_ori down | Normal learning | Continue training |
| Reward down + time_out down + bad_ori up | Policy collapse | Consider rollback to earlier checkpoint |
| Reward down + time_out stable + bad_ori stable | Reward weight change | Check if curriculum changed difficulty |
| Entropy drops sharply + policy_std < 0.01 | Overfitting | Reduce learning rate or increase entropy bonus |
| Value_loss spikes (>100) | Value function divergence | Learning rate too large or data anomaly |

---

## Appendix C: num_envs Performance Analysis

> Hardware: 4x RTX 6000D (85.7 GB each), Isaac Lab 0.47.2, Isaac Sim 4.5.0
> Task: MagicBot Z1 12DOF Locomotion, s4_full_terrain

### Core Formula

```
Timesteps per iteration = num_envs_per_gpu x num_gpus x num_steps(=24)
Total training volume   = max_iterations x num_envs_per_gpu x num_gpus x num_steps
```

### Core Comparison Table

| Metric | 4096/GPU | 16384/GPU | 32768/GPU |
|--------|----------|-----------|-----------|
| Total parallel envs | 16,384 | 65,536 | 131,072 |
| Timesteps per iter | 393k | 1,573k | 3,146k |
| Measured iter time | ~2.2s | ~4.5s* | ~6.0s |
| Throughput (steps/s) | ~176k | ~330k* | ~524k |
| Throughput multiplier | 1.0x | ~1.9x | 3.0x |
| GPU VRAM usage | ~6-8 GB | ~12-15 GB* | ~20.7 GB |
| GPU utilization | ~8% | ~15%* | ~24% |
| GPU remaining | ~78 GB | ~71 GB* | ~65 GB |
| 55k iters total timesteps | 21.6B | 86.5B | 173B |
| Equiv. timesteps for 55k iters | 1x | 4x | 8x |
| Iters for equal timesteps | 55,000 | 13,750 | 6,875 |
| ETA for equal timesteps | ~34h | ~17h* | ~11.5h |

> *16384/GPU values estimated from interpolation of 4096 and 32768 measurements

### Key Findings

1. **Non-linear throughput**: 8x envs yields only 3x throughput. PPO collection phase is the serial bottleneck -- GPU compute is not the limit, memory bandwidth and communication are.

2. **GPU underutilization**: Even at 32768 envs, only 24% of 85 GB VRAM is used. RTX 6000D capacity is far from fully utilized; increasing envs further yields diminishing returns.

3. **max_iterations must scale with envs**: Without adjustment, 32768 envs at 55k iters covers 8x the training volume, causing wasted compute, mismatched LR schedule, and potential overfitting.

4. **Training quality**: More envs means richer experience per PPO update, more accurate gradient estimates, and more stable training.

### Recommendation

| Scenario | Recommended | Reason |
|----------|------------|--------|
| Quick experiment / debug | 4096/GPU | Fast startup, low VRAM |
| **Production training (4 GPU)** | **16384/GPU** | **Best cost/benefit, ample VRAM** |
| Fastest convergence | 32768/GPU | Highest throughput, must adjust max_iterations |
| Stress test | 65536/GPU | May approach VRAM limit, severe diminishing returns |

> Note: When using 16384 or 32768, proportionally reduce `max_iterations` or training volume will far exceed expectations.

---

## Appendix D: Implicit vs Explicit PD Control

### PD Control Basics

Both implicit and explicit modes share the same control goal: drive the joint to a target position.

```
torque = kp x (target_pos - current_pos) + kd x (target_vel - current_vel)
```

- `kp` (stiffness): spring stiffness; larger position error produces larger torque
- `kd` (damping): damping coefficient; faster motion produces stronger braking force
- `torque`: final torque applied to joint

The difference between the two modes is not the formula, but **when and by whom the formula is computed**.

---

### Implicit PD (ImplicitActuator) -- Current Configuration

**Who computes**: PhysX physics engine internally.

```
Policy outputs action
    -> action converted to joint target position
    -> target position + kp/kd written to PhysX JointDrive property
    -> PhysX "implicitly" integrates PD torque into physics solve
    -> Final joint motion (you never see intermediate torque values)
```

**"Implicit" meaning**: Torque is not computed separately. PhysX treats PD control as a physical constraint, solving it together with collisions and gravity. PD torque and physics integration are coupled -- you cannot separate it from the simulation.

Analogy: You tell a spring "go to this position", but the spring force is not computed by you -- it is naturally produced by the physics engine during simulation.

**Characteristics**:
- Numerically stable even at large time steps
- Torque is invisible: you cannot know actual torque applied
- No hard clipping: physics engine may produce forces exceeding motor limits
- Inconsistent with real motor behavior

---

### Explicit PD (IdealPDActuator) -- Recommended for Sim2Real

**Who computes**: Python code, before each physics simulation step.

```
Policy outputs action
    -> action converted to joint target position
    -> Python explicitly computes: torque = kp x error + kd x error_dot
    -> torque clipped by effort_limit (excess above 120Nm is discarded)
    -> Clipped torque applied as external force to joint
    -> PhysX handles only physics response (does not participate in PD computation)
```

**"Explicit" meaning**: Torque is computed separately and clearly. Before each simulation step, the formula produces a torque value, which is clipped and then handed to the physics engine. PD control and physics simulation are decoupled.

Analogy: You calculate exactly how much force is needed, then hand that force to the physics engine to execute.

**Characteristics**:
- Torque fully controllable: you know exactly what was applied, and can clip it
- Consistent with MuJoCo (which also uses explicit PD)
- Consistent with real motors (real motor controllers compute PD this way)
- Suitable for sim2real: simulation and reality control logic are fully aligned

---

### Door Analogy

Imagine controlling the angle of a door:

| | Implicit | Explicit |
|---|---|---|
| Approach | Mount a spring on the door, set the spring endpoint to target, let physics engine simulate | Push the door directly, you calculate the force needed and apply it |
| What you see | Only see the door moving, cannot see spring force magnitude | You know exactly how much force you applied, can limit maximum |
| Exceeding motor capacity | Spring may produce force beyond motor limits (physics engine does not care) | You check force after computing, clip if over limit |
| Real motor correspondence | Real motors are not springs, cannot map directly | Real motor controllers work exactly this way |

---

### Sim2Real Impact

**Why implicit PD causes sim2sim gap**:

1. **PhysX PD behavior != MuJoCo PD behavior**: Implicit mode integrates PD into constraint solving, producing different behavior than explicit PD. MuJoCo uses explicit PD. Strategies learned under PhysX implicit PD that rely on high-frequency compensation become completely inapplicable under MuJoCo explicit PD.

2. **Torque limiting behavior differs**: Implicit mode may "soft" exceed limits (actual torque exceeds effort_limit). Explicit/real hardware: hard clipping, excess is directly discarded.

3. **High-frequency response differs**: Implicit PD is inside physics integration, response is faster and smoother. Explicit PD has a one-step delay. Real motors also have delay.

**Why explicit PD improves sim2real**:

```
Isaac Lab (explicit PD)  ~=  MuJoCo (explicit PD)  ~=  Real motor (explicit PD)
         |                          |                          |
   Same control logic         Same control logic         Same control logic
```

All three compute PD torque the same way, so learned behavior transfers directly.

### Migration Cost

The code change is small (3 locations in `magiclab.py`), but requires **retraining from scratch** because:
- Policy trained under implicit PD has adapted to PhysX-specific dynamics
- Explicit PD joint response will differ (may be "softer" due to clipping)
- Policy must relearn how to balance under explicit PD

### Reference

- Current config: `source/magiclab_rl_lab/magiclab_rl_lab/assets/robots/magiclab.py`
- Isaac Lab Actuator docs: `isaaclab.actuators.IdealPDActuatorCfg`
