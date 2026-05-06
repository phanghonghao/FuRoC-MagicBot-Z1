# magiclab_rl_lab 框架详解

> 基于路径 `magiclab_rl_lab/source/magiclab_rl_lab/magiclab_rl_lab/` 的完整框架解析

---

## 1. 目录结构

```
magiclab_rl_lab/
├── __init__.py                          # 空，顶层包
├── assets/
│   └── robots/
│       └── magiclab.py                  # 机器人 URDF/USD 配置 + Actuator 配置
├── data/
│   └── robots/
│       └── magicbot-Z1/
│           ├── meshes/                  # STL 网格文件
│           └── urdf/                    # 3 个 URDF 变体 (12dof, 23dof, arm_ready_pos)
├── tasks/
│   └── locomotion/
│       ├── __init__.py                  # 注册 Gym 环境
│       ├── agents/
│       │   ├── __init__.py
│       │   └── rsl_rl_ppo_cfg.py       # PPO 超参数配置
│       ├── mdp/                         # ★ 核心 MDP 组件
│       │   ├── __init__.py              # 转导所有 MDP 函数
│       │   ├── commands/
│       │   │   ├── __init__.py
│       │   │   └── velocity_command.py  # 自定义速度指令
│       │   ├── curriculums.py           # 课程学习
│       │   ├── observations.py          # 自定义观测函数
│       │   └── rewards.py              # 自定义奖励函数
│       └── robots/
│           ├── __init__.py              # 导入机器人子模块
│           └── z1/
│               ├── __init__.py          # Gym 环境注册入口
│               └── 12dof/
│                   ├── __init__.py
│                   └── velocity_env_cfg.py  # ★ 主环境配置文件
├── ui_extension_example.py              # Omniverse UI 扩展示例
└── utils/
    ├── export_deploy_cfg.py             # 部署配置导出器
    └── parser_cfg.py                    # 配置解析工具
```

---

## 2. 模块依赖关系

```
                    ┌─────────────────┐
                    │  Isaac Lab SDK  │
                    │  (isaaclab.*)   │
                    └────────┬────────┘
                             │ 提供 base classes
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼─────┐  ┌────▼─────┐  ┌────▼──────────┐
    │ assets/       │  │ tasks/   │  │ utils/        │
    │ robots/       │  │ mdp/     │  │ export_deploy │
    │ magiclab.py   │  │ rewards  │  │ parser_cfg    │
    └───────┬───────┘  │ obs      │  └───────────────┘
            │          │ cmd      │
            │          │ curr     │
            │          └────┬─────┘
            │               │
            └───────┬───────┘
                    │ 引用
          ┌─────────▼──────────┐
          │ velocity_env_cfg.py│  ← 主配置文件，组装一切
          └────────────────────┘
                    │ 注册为
          ┌─────────▼──────────┐
          │ Gym Environment    │
          │ "Magiclab-Z1-     │
          │  12dof-Velocity"   │
          └────────────────────┘
```

---

## 3. 机器人配置 (`assets/robots/magiclab.py`)

### 3.1 类继承体系

```
sim_utils.UrdfFileCfg
    └── MagiclabUrdfFileCfg        # 自定义 URDF 加载配置
            ├── fix_base = False           # 不固定基座（浮动基）
            ├── activate_contact_sensors   # 启用接触传感器
            ├── replace_cylinders_with_capsules = True
            ├── joint_drive: Kp=0, Kd=0    # URDF 内置增益设为 0（由 IsaacLab actuator 接管）
            ├── articulation_props: self_collision=True, pos_iter=8, vel_iter=4
            └── rigid_props: 无阻尼，最大速度 1000

ArticulationCfg (IsaacLab)
    └── MagiclabArticulationCfg    # 自定义关节配置
            ├── joint_sdk_names: list[str]   # SDK 端关节名列表（用于部署映射）
            └── soft_joint_pos_limit_factor = 0.9
```

### 3.2 Z1 12-DoF 具体配置

```python
MAGICLAB_Z1_12DOF_CFG = MagiclabArticulationCfg(
    spawn=MagiclabUrdfFileCfg(
        asset_path=".../MagicBotZ1_12dof_arm_ready_pos.urdf",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.69),       # 初始高度 0.69m
        joint_pos={
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_pitch_joint": -0.35,   # 微微前倾
            ".*_knee_joint": 0.7,           # 膝盖弯曲
            ".*_ankle_pitch_joint": -0.35,  # 脚踝补偿
            ".*_ankle_roll_joint.*": 0.0,
        },
    ),
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_roll", ".*_hip_yaw", ".*_hip_pitch", ".*_knee"],
            effort_limit=120, velocity_limit=20,
            stiffness={"hip": 100, "knee": 150},    # Nm/rad
            damping={"hip": 4, "knee": 5},            # Nm·s/rad
            armature={"hip": 0.02863, "knee": 0.02863},
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch", ".*_ankle_roll"],
            effort_limit=50, velocity_limit=15,
            stiffness=60, damping=3,
            armature=0.01503,
        ),
    },
    joint_sdk_names=[              # SDK 端关节顺序（部署用）
        "left_hip_pitch_joint",    # idx 0
        "left_hip_roll_joint",     # idx 1
        "left_hip_yaw_joint",      # idx 2
        "left_knee_joint",         # idx 3
        "left_ankle_pitch_joint",  # idx 4
        "left_ankle_roll_joint",   # idx 5
        "right_hip_pitch_joint",   # idx 6
        "right_hip_roll_joint",    # idx 7
        "right_hip_yaw_joint",     # idx 8
        "right_knee_joint",        # idx 9
        "right_ankle_pitch_joint", # idx 10
        "right_ankle_roll_joint",  # idx 11
    ],
)
```

### 3.3 Actuator 模型：IdealPDActuator

使用**显式 PD 控制**，力矩公式：

```
τ = Kp × (q_desired - q_actual) - Kd × qdot_actual
```

与 `ImplicitActuator` 的区别：
| | IdealPD (显式) | Implicit (隐式) |
|---|---|---|
| 公式 | τ = Kp·Δq - Kd·q̇ | 由 PhysX 求解器内部计算 |
| 精度 | 精确力矩输出 | 数值积分近似 |
| Sim2Real | **可直接映射到 MuJoCo** | 需要额外转换 |
| 稳定性 | 需要更高控制频率 | 更宽容 |

---

## 4. 主环境配置 (`velocity_env_cfg.py`)

这是整个训练管线的核心配置文件，定义了完整的 MDP (马尔可夫决策过程)。

### 4.1 场景 (`RobotSceneCfg`)

```python
scene: RobotSceneCfg = RobotSceneCfg(num_envs=16384, env_spacing=2.5)
```

| 组件 | 说明 |
|------|------|
| `terrain` | 程序化地形生成器 (8×8m 网格, 50% 平坦) |
| `robot` | Z1 12DoF 机器人 |
| `height_scanner` | RayCaster, 从骨盆上方 20m 发射射线，网格分辨率 0.1m |
| `contact_forces` | 接触力传感器 (history=3, track_air_time=True) |
| `sky_light` | HDR 环境光照 |

### 4.2 动作空间 (`ActionsCfg`)

```python
JointPositionAction(
    joint_names=[12个关节名],    # 有序列表
    scale=0.25,                   # 网络输出 × 0.25 = 关节位置偏移 (rad)
    use_default_offset=True,      # 偏移量叠加在默认关节位置上
    preserve_order=True,          # 保持关节顺序与 SDK 一致
)
```

**动作维度**: 12 (对应 12 个关节)
**动作范围**: 网络输出 [-1, 1] → 实际偏移 [-0.25, 0.25] rad

### 4.3 观测空间 (`ObservationsCfg`)

#### Policy 观测 (Actor 网络)

| 观测项 | 函数 | 维度 | Scale | Noise | 说明 |
|--------|------|------|-------|-------|------|
| `base_ang_vel` | `base_ang_vel` | 3 | 0.2 | ±0.3 | 角速度 |
| `projected_gravity` | `projected_gravity` | 3 | — | ±0.15 | 重力投影 |
| `velocity_commands` | `generated_commands` | 3 | — | — | 速度指令 |
| `joint_pos_rel` | `joint_pos_rel` | 12 | — | ±0.05 | 关节相对位置 |
| `joint_vel_rel` | `joint_vel_rel` | 12 | 0.05 | ±2.0 | 关节相对速度 |
| `last_action` | `last_action` | 12 | 1.0 | — | 上一步动作 |
| `gait_phase` | `gait_phase` | 2 | — | — | 步态相位 |

**单帧 Policy 维度**: 3+3+3+12+12+12+2 = **47**
**History 长度**: 5 → Policy 总维度: 47 × 5 = **235**

#### Critic 观测 (价值网络，特权信息)

| 观测项 | 维度 | 说明 |
|--------|------|------|
| `base_lin_vel` | 3 | 线速度 (无噪声) |
| `base_ang_vel` | 3 | 角速度 (无噪声) |
| `projected_gravity` | 3 | 重力投影 (无噪声) |
| `velocity_commands` | 3 | 速度指令 |
| `joint_pos_rel` | 12 | 关节位置 |
| `joint_vel_rel` | 12 | 关节速度 |
| `last_action` | 12 | 上一步动作 |
| `gait_phase` | 2 | 步态相位 |
| `contact_mask` | 2 | 脚接触掩码 |

**单帧 Critic 维度**: 3+3+3+3+12+12+12+2+2 = **52**
**History 长度**: 5 → Critic 总维度: 52 × 5 = **260**

### 4.4 指令 (`CommandsCfg`)

```python
UniformLevelVelocityCommandCfg(
    resampling_time_range=(10.0, 10.0),  # 每 10s 重新采样
    rel_standing_envs=0.02,               # 2% 环境保持站立
    rel_heading_envs=1.0,                 # 100% 随机朝向
    heading_command=False,                # 不使用朝向指令
    ranges:                               # 初始指令范围 (课程学习起点)
        lin_vel_x: (-0.1, 0.1)
        lin_vel_y: (-0.1, 0.1)
        ang_vel_z: (-0.1, 0.1)
    limit_ranges:                         # 最大指令范围 (课程学习终点)
        lin_vel_x: (-0.5, 1.0)
        lin_vel_y: (-0.5, 0.5)
        ang_vel_z: (-0.5, 0.5)
)
```

### 4.5 奖励函数 (`RewardsCfg`)

#### 任务奖励 (正)

| 名称 | 函数 | 权重 | 说明 |
|------|------|------|------|
| `track_lin_vel_xy` | `track_lin_vel_xy_yaw_frame_exp` | 1.0 | 跟踪线速度 (yaw frame, exp 核) |
| `track_ang_vel_z` | `track_ang_vel_z_exp` | 0.5 | 跟踪角速度 (exp 核) |
| `alive` | `is_alive` | 0.15 | 存活奖励 |
| `feet_contact_number` | `feet_contact_number` | 0.5 | 步态接触一致性 |
| `feet_clearance` | `foot_clearance_reward` | 1.0 | 摆动脚离地高度 |

#### 惩罚 (负)

| 名称 | 函数 | 权重 | 说明 |
|------|------|------|------|
| `base_linear_velocity` | `lin_vel_z_l2` | -2.0 | 垂直速度惩罚 |
| `base_angular_velocity` | `ang_vel_xy_l2` | -0.05 | 水平角速度惩罚 |
| `joint_vel` | `joint_vel_l2` | -0.001 | 关节速度惩罚 |
| `joint_acc` | `joint_acc_l2` | -5e-7 | 关节加速度惩罚 |
| `action_rate` | `action_rate_l1` | -0.1 | 动作变化率惩罚 |
| `dof_pos_limits` | `joint_pos_limits` | -5.0 | 关节极限惩罚 |
| `energy` | `energy` | -2e-5 | 能量消耗惩罚 |
| `joint_deviation_legs` | `joint_deviation_l1` | -0.7 | hip_roll/hip_yaw 偏离默认 |
| `flat_orientation_l2` | `flat_orientation_l2` | -5.0 | 身体倾斜惩罚 |
| `base_height` | `base_height_l2` | -10.0 | 高度偏离 0.7m |
| `stand_still` | `stand_still_joint_deviation_l1` | -3.5 | 站立时关节偏离 |
| `feet_slide` | `feet_slide` | -0.2 | 脚滑惩罚 |
| `undesired_contacts` | `undesired_contacts` | -1.0 | 非脚部位接触 |

### 4.6 终止条件 (`TerminationsCfg`)

| 条件 | 函数 | 参数 | 说明 |
|------|------|------|------|
| `time_out` | `time_out` | 20s | 到达最大 episode 长度 |
| `base_height` | `root_height_below_minimum` | 0.2m | 基座高度过低 |
| `bad_orientation` | `bad_orientation` | 0.8 rad | 倾斜角过大 (~46°) |

### 4.7 事件 / Domain Randomization (`EventCfg`)

| 时机 | 事件 | 参数 | 说明 |
|------|------|------|------|
| **startup** | `physics_material` | friction (0.1, 2.0) | 随机化摩擦系数 |
| **startup** | `add_base_mass` | mass_scale (0.5, 1.5) | 随机化骨盆质量 |
| **startup** | `randomize_rigid_body_mass_others` | mass_scale (0.5, 1.5) | 随机化所有连杆质量 |
| **reset** | `base_external_force_torque` | 0 | 清除外力 |
| **reset** | `reset_base` | pos ±0.5m, vel ±0.5 | 随机化初始位姿 |
| **reset** | `reset_robot_joints` | pos=1.0, vel ±1.0 | 随机化关节初始状态 |
| **interval** | `push_robot` | 3-5s, vel ±1.0 m/s | 随机推力扰动 |

### 4.8 课程学习 (`CurriculumCfg`)

| 课程 | 函数 | 机制 |
|------|------|------|
| `terrain_levels` | `terrain_levels_vel` | 随 reward 提升地形难度 |
| `lin_vel_cmd_levels` | `lin_vel_cmd_levels` | reward > 0.8×weight 时扩大速度指令范围 ±0.1 |
| `ang_vel_cmd_levels` | `ang_vel_cmd_levels` | reward > 0.8×weight 时扩大角速度指令范围 ±0.1 |

### 4.9 仿真参数

```python
decimation = 10              # 控制频率 = 1/(dt×decimation) = 1/(0.002×10) = 50 Hz
episode_length_s = 20.0      # 每个 episode 20 秒
sim.dt = 0.002               # 物理步长 2ms (500 Hz)
```

---

## 5. 自定义 MDP 函数详解

### 5.1 自定义奖励函数 (`mdp/rewards.py`)

#### `energy(env, asset_cfg)`
```
reward = Σ |qvel_i| × |torque_i|
```
惩罚所有关节的机械能消耗。

#### `stand_still(env, command_threshold, command_name)`
```
reward = Σ |joint_pos - default_pos| × (cmd_norm < threshold)
```
当速度指令接近零时，惩罚关节偏离默认位置。

#### `foot_clearance_reward(env, asset_cfg, target_height, std, tanh_mult)`
```
error = (foot_z - target_height)²
vel_factor = tanh(tanh_mult × |foot_vel_xy|)
reward = exp(-Σ(error × vel_factor) / std)
```
奖励摆动脚在移动时离地达到 target_height (0.1m)，静止时 (vel_factor→0) 不惩罚。

#### `feet_gait(env, period, offset, sensor_cfg, threshold)`
```
phase = (time % period / period + offset) % 1.0
is_stance = phase < threshold
reward = Σ ~(is_stance XOR is_contact)
```
奖励接触状态与期望步态相位一致。

#### `feet_contact_number(env, period, sensor_cfg)`
```
sin_phase = sin(2π × time/period)
stance_mask[left] = sin_phase ≥ 0    # 左脚支撑相
stance_mask[right] = sin_phase < 0   # 右脚支撑相
contact = |force_z| > 5
reward = mean(where(contact == stance_mask, 1.0, -0.3))
```
基于正弦步态相位的接触一致性奖励。静止时 (cmd<0.02) 双脚都标记为支撑。

#### `joint_mirror(env, asset_cfg, mirror_joints)`
```
reward = (1/N) × Σ (joint_pos[left_i] - joint_pos[right_i])²
```
惩罚左右对称关节的位置差异，鼓励对称步态。

#### `feet_stumble(env, sensor_cfg)`
```
reward = any(|force_xy| > 4 × |force_z|)
```
检测脚碰到垂直面（水平力远大于垂直力）。

#### `air_time_variance_penalty(env, sensor_cfg)`
```
reward = var(air_time) + var(contact_time)
```
惩罚双脚空中/接触时间差异，鼓励均匀步态。

### 5.2 自定义观测函数 (`mdp/observations.py`)

#### `gait_phase(env, period) → shape (num_envs, 2)`

返回步态相位掩码：
```
sin_phase = sin(2π × (time % period) / period)
stance_mask[:, 0] = sin_phase ≥ 0   # 左脚是否在支撑相
stance_mask[:, 1] = sin_phase < 0   # 右脚是否在支撑相
```
静止时 (cmd_norm < 0.02) 返回 [1, 1]（双脚支撑）。

#### `contact_mask(env, sensor_cfg) → shape (num_envs, num_bodies)`

```
mask = |net_forces_w[:, body_ids, 2]| > 5
```
返回每个脚的接触布尔掩码 (z 方向力 > 5N)。

### 5.3 自定义课程学习 (`mdp/curriculums.py`)

#### `lin_vel_cmd_levels(env, env_ids, reward_term_name)`

每个 episode 结束时检查：
- 如果平均 reward > 0.8 × weight → 扩大 lin_vel_x/y 范围 ±0.1
- 范围被 clamp 到 limit_ranges 内

#### `ang_vel_cmd_levels(env, env_ids, reward_term_name)`

同上，但控制 ang_vel_z 范围。

### 5.4 自定义指令 (`mdp/commands/velocity_command.py`)

#### `UniformLevelVelocityCommandCfg`

继承自 IsaacLab 的 `UniformVelocityCommandCfg`，添加：
```python
limit_ranges: Ranges   # 最大指令范围，用于课程学习
```

---

## 6. PPO 训练配置 (`agents/rsl_rl_ppo_cfg.py`)

### 6.1 网络结构

```python
policy = RslRlPpoActorCriticCfg(
    init_noise_std=1.0,
    actor_hidden_dims=[512, 256, 128],    # 3 层 MLP
    critic_hidden_dims=[512, 256, 128],    # 3 层 MLP
    activation="elu",
)
```

**Actor 输入**: 235 (policy obs × history=5)
**Actor 输出**: 12 (joint position offsets)
**Critic 输入**: 260 (critic obs × history=5)
**Critic 输出**: 1 (state value)

### 6.2 PPO 超参数

| 参数 | 值 | 说明 |
|------|------|------|
| `num_steps_per_env` | 24 | 每次 rollout 的步数 |
| `max_iterations` | 50000 | 最大训练迭代 |
| `save_interval` | 100 | 每 100 iter 保存 |
| `learning_rate` | 1e-3 | 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.95 | GAE lambda |
| `clip_param` | 0.2 | PPO clip 范围 |
| `entropy_coef` | 0.01 | 熵正则系数 |
| `desired_kl` | 0.01 | 目标 KL 散度 (adaptive schedule) |
| `num_learning_epochs` | 5 | 每次更新的 epoch 数 |
| `num_mini_batches` | 4 | mini-batch 数量 |
| `max_grad_norm` | 1.0 | 梯度裁剪 |
| `empirical_normalization` | False | 不使用经验归一化 |

### 6.3 训练吞吐量估算

```
每 iter 样本数 = num_envs × num_steps_per_env = 16384 × 24 = 393,216
每 iter 更新量 = 5 epochs × 4 mini_batches = 20 次梯度更新
总样本数 ≈ 50,000 × 393,216 ≈ 19.6B transitions
```

---

## 7. 工具模块 (`utils/`)

### 7.1 `export_deploy_cfg.py`

训练完成后自动调用，导出 `deploy.yaml`：

```yaml
joint_ids_map: [0, 1, 2, ...]        # 关节顺序映射
step_dt: 0.02                          # 控制周期 (50Hz)
stiffness: [100, 100, 100, 150, ...]   # Kp
damping: [4, 4, 4, 5, ...]            # Kd
default_joint_pos: [...]               # 默认关节位置
commands:
  base_velocity:
    ranges: {lin_vel_x: [...], ...}
actions:
  JointPositionAction:
    scale: [0.25 × 12]
    offset: [...]                       # 默认位置偏移
    joint_ids: [0, 1, 2, ...]
observations:
  base_ang_vel: {scale: [0.2×3], history_length: 5}
  ...
```

### 7.2 `parser_cfg.py`

解析 Gym 注册表中的环境配置，支持覆盖：
- `device`: GPU 设备
- `num_envs`: 环境数量
- `use_fabric`: 是否使用 Fabric 接口

---

## 8. 数据流

```
┌─────────────────────────────────────────────────────────┐
│                     Training Loop                        │
│                                                          │
│  ┌──────────┐    ┌───────────┐    ┌───────────────────┐ │
│  │  IsaacLab │    │  MDP Env  │    │   RSL-RL PPO     │ │
│  │  PhysX    │───▶│           │───▶│                   │ │
│  │  Sim      │    │ obs+reward│    │ actor(π) + critic │ │
│  └──────────┘    └───────────┘    └───────────────────┘ │
│       ▲               │                     │           │
│       │               │              action (12d)       │
│       └───────────────┴─────────────────────┘           │
│                                                          │
│  仿真: PhysX 500Hz (dt=2ms)                             │
│  控制: 50Hz (decimation=10)                              │
│  Episode: 20s = 1000 步                                  │
└─────────────────────────────────────────────────────────┘
```

### 单步数据流：

```
1. 网络输出 action ∈ [-1, 1]^12
2. action × 0.25 = offset ∈ [-0.25, 0.25]^12 rad
3. target_pos = default_pos + offset
4. IdealPDActuator 计算:
   τ = Kp × (target - actual) - Kd × actual_vel
5. PhysX 积分一步 (×10 步 = 一个控制周期)
6. 采集观测 → 加噪 → 拼接 history → 输入网络
```

---

## 9. MDP 函数来源汇总

`mdp/__init__.py` 的导入策略：

```python
from isaaclab.envs.mdp import *                          # IsaacLab 基础 MDP
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # IsaacLab 速度跟踪 MDP
from .commands import *                                   # 自定义指令
from .curriculums import *                                # 自定义课程
from .observations import *                               # 自定义观测
from .rewards import *                                    # 自定义奖励
```

这意味着项目可用的 MDP 函数 = IsaacLab 内置 + IsaacLab Locomotion + 自定义。

### IsaacLab 内置函数（部分常用）

| 类别 | 函数 | 来源 |
|------|------|------|
| 奖励 | `track_lin_vel_xy_yaw_frame_exp`, `track_ang_vel_z_exp` | locomotion |
| 奖励 | `lin_vel_z_l2`, `ang_vel_xy_l2`, `joint_vel_l2`, `joint_acc_l2` | isaaclab.envs.mdp |
| 奖励 | `action_rate_l1`, `joint_pos_limits`, `joint_deviation_l1` | isaaclab.envs.mdp |
| 奖励 | `flat_orientation_l2`, `base_height_l2` | isaaclab.envs.mdp |
| 奖励 | `feet_slide`, `undesired_contacts` | locomotion |
| 奖励 | `is_alive`, `stand_still_joint_deviation_l1` | isaaclab.envs.mdp |
| 观测 | `base_ang_vel`, `base_lin_vel`, `projected_gravity` | isaaclab.envs.mdp |
| 观测 | `joint_pos_rel`, `joint_vel_rel`, `last_action` | isaaclab.envs.mdp |
| 观测 | `generated_commands` | isaaclab.envs.mdp |
| 动作 | `JointPositionAction` | isaaclab.envs.mdp |
| 事件 | `randomize_rigid_body_material`, `randomize_rigid_body_mass` | isaaclab.envs.mdp |
| 事件 | `push_by_setting_velocity`, `reset_root_state_uniform` | isaaclab.envs.mdp |
| 事件 | `reset_joints_by_scale`, `apply_external_force_torque` | isaaclab.envs.mdp |
| 终止 | `time_out`, `root_height_below_minimum`, `bad_orientation` | isaaclab.envs.mdp |
| 课程 | `terrain_levels_vel` | locomotion |

---

## 10. 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| Actuator 模型 | `IdealPDActuator` (显式 PD) | Sim2Real 一致性，可直接映射 MuJoCo |
| 动作空间 | 关节位置偏移 | 比力矩控制更稳定，比绝对位置更灵活 |
| 观测 History | 5 帧 | 捕获动态信息 (速度、加速度) |
| 关节顺序 | `preserve_order=True` | 确保训练/部署关节数组对齐 |
| 课程学习 | 速度指令 + 地形 | 从简单到困难渐进 |
| 步态编码 | 正弦相位 | 简单高效，周期 0.6s ≈ 1.67Hz 步频 |
| Domain Rand | 摩擦/质量/推力 | 覆盖 Sim2Real 差异 |
