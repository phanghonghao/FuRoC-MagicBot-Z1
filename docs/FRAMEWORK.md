# MagicBot Z1 — RL Locomotion Framework

> MagicBot Z1 12DOF 双足机器人强化学习运动训练框架完整技术文档。

---

## 1. 项目总览

```
magiclab_rl_lab/
├── source/magiclab_rl_lab/magiclab_rl_lab/   # 核心源码包
│   ├── tasks/locomotion/                     # 任务定义（环境、MDP、Agent）
│   ├── assets/robots/                        # 机器人 Articulation 配置
│   ├── data/robots/                          # URDF / Mesh 文件
│   └── utils/                                # 工具函数
├── scripts/                                  # 训练/评估/导出脚本
│   ├── rsl_rl/                               # RSL-RL 训练与 Play 脚本
│   └── automation/                           # 自动化多阶段训练编排
├── sim2sim/                                  # MuJoCo Sim2Sim 验证
├── deploy/                                   # 真机部署
├── training_plans/                           # 多阶段训练计划 YAML
├── train_bash.sh                             # 训练启动脚本
├── play_bash.sh                              # 评估启动脚本
└── logs/                                     # 训练日志与 Checkpoint
```

### 三平台架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    RTX 6000D (远程训练服务器)                      │
│  85GB VRAM · 4096 envs · Isaac Lab 0.47.2 + rsl_rl              │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ 训练引擎      │  │ 模型管理      │  │ 自动化 Pipeline       │  │
│  │ Isaac Lab    │  │ checkpoint   │  │ orchestrator          │  │
│  │ rsl_rl PPO   │  │ best_model   │  │ train_monitor         │  │
│  └──────┬───────┘  └──────┬───────┘  └───────────────────────┘  │
│         │                 │                                      │
│  logs/rsl_rl/magiclab_z1_12dof_velocity/{run_dir}/             │
└─────────┼─────────────────┼──────────────────────────────────────┘
          │ JIT/ONNX export │ model_*.pt + deploy.yaml
          ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                  DGX Spark (录制 / 备用训练)                      │
│  ARM64 · Isaac Lab 0.54.3 · Isaac Sim 5.1                       │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │ spark_play   │  │ MuJoCo       │                             │
│  │ JIT-only 评估 │  │ Sim2Sim 验证 │                             │
│  └──────────────┘  └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
          │ .mp4 videos
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                本地 Windows (整理 / 分析)                         │
│  D:\Desktop_Files\GPU-Train\RTX6000\Magicbot_Z1\               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ videos/      │  │ docs/        │  │ /gpu-train   │          │
│  │ 版本视频归档  │  │ 文档分析      │  │ CLI 管理命令  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### 各平台目录详解

#### 远程 RTX 服务器 (`phh@192.168.120.155`)

```
~/
├── magiclab_rl_lab/                    # 主代码仓库
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
├── IsaacLab/                           # Isaac Lab 框架 (v0.47.2)
├── magicbot-z1_description/            # URDF + MJCF + meshes
├── mujoco_record_video.py             # MuJoCo EGL 离屏录像
└── miniconda3/envs/isaaclab/          # Conda 环境
```

#### 远程 Spark 服务器 (`ssh spark`)

```
~/
├── magiclab_rl_lab/                    # 与 RTX 同步的代码
│   ├── scripts/spark_play.py           # Spark JIT-only 播放
│   └── logs/rsl_rl/.../exported/policy.pt
└── miniconda3/envs/env_isaaclab/       # Conda 环境 (ARM64)
```

#### 本地 Windows (`D:\Desktop_Files\GPU-Train\RTX6000\`)

```
RTX6000/
├── Magicbot_Z1/
│   ├── magiclab_rl_lab/               # 代码镜像
│   ├── magicbot-z1_description/       # 机器人描述镜像
│   ├── magicbot-z1_sdk/               # SDK (ARM64 + x86_64)
│   ├── configs/                       # 环境配置备份
│   ├── scripts/ (compare_videos, label_video)
│   ├── docs/                          # 文档
│   ├── videos/                        # 下载的录像
│   ├── best_models.json              # scp 下来的最佳模型汇总
│   └── IsaacLab/
├── spark_deploy.sh                    # 一键部署到 Spark
├── launch_dual_training.sh
└── videos/
```

> **注意**: 训练 checkpoint (`model_*.pt`) 只存在于远程 RTX，本地是代码副本和下载的结果文件。

---

## 2. 源码架构

### 2.1 模块依赖关系

```
tasks/locomotion/
├── __init__.py                          # 导入 robots
├── agents/
│   ├── __init__.py
│   └── rsl_rl_ppo_cfg.py               # PPO 超参数配置
├── mdp/
│   ├── __init__.py                      # 导入 Isaac Lab mdp
│   ├── observations.py                  # 自定义观测函数（gait_phase, contact_mask）
│   ├── rewards.py                       # 自定义奖励函数（energy, feet_gait, etc.）
│   ├── curriculums.py                   # 课程学习（速度命令递进）
│   └── commands/
│       └── velocity_command.py          # 带 limit_ranges 的速度命令
└── robots/
    ├── __init__.py                      # 导入 z1
    └── z1/
        ├── __init__.py                  # 导入 12dof
        └── 12dof/
            ├── __init__.py              # Gym 任务注册
            └── velocity_env_cfg.py      # 环境完整配置
```

### 2.2 任务注册

`robots/z1/12dof/__init__.py` 通过 Gymnasium 注册任务：

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

## 3. 环境配置

### 3.1 场景 (RobotSceneCfg)

| 组件 | 配置 |
|------|------|
| 地形 | `TerrainImporterCfg`，生成器模式，50% flat |
| 地形尺寸 | 8m × 8m，9 行 × 21 列 |
| 地面摩擦 | static=1.0, dynamic=1.0 |
| 机器人 | `MAGICLAB_Z1_12DOF_CFG`，URDF: `MagicBotZ1_12dof_arm_ready_pos.urdf` |
| 初始高度 | 0.69m |
| 接触传感器 | 全身 `.*`，history_length=3，track_air_time=True |
| 高度扫描 | RayCaster on pelvis，Grid 1.6×1.0m，分辨率 0.1m |
| 并行环境 | 默认 4096，间距 2.5m |

### 3.2 仿真参数

| 参数 | 值 |
|------|------|
| 物理步长 (dt) | 0.002s (500Hz) |
| Decimation | 10 → 控制频率 50Hz |
| Episode 长度 | 20s (1000 步) |
| 渲染间隔 | = decimation |

### 3.3 观测空间 (235-dim)

**Policy 观测 = 47-dim/帧 × 5 帧 history_length**

Isaac Lab 使用 **per-term interleaved** 布局（非 frame-stacked）：
```
[ang_vel×5, gravity×5, cmd×5, jpos×5, jvel×5, act×5, gait×5]
```

| 观测项 | 维度 | 缩放 | 噪声 |
|--------|------|------|------|
| `base_ang_vel` | 3 | ×0.2 | U(−0.2, 0.2) |
| `projected_gravity` | 3 | — | U(−0.1, 0.1) |
| `velocity_commands` | 3 | — | — |
| `joint_pos_rel` | 12 | — | U(−0.02, 0.02) |
| `joint_vel_rel` | 12 | ×0.05 | U(−1.5, 1.5) |
| `last_action` | 12 | ×1.0, clip(−100, 100) | — |
| `gait_phase` | 2 | — | — |

**Critic 额外观测** (同 5 帧历史)：
- `base_lin_vel` (3) — 真实线速度（特权信息）
- `contact_mask` (2) — 脚踝接触力 > 5N 的二值 mask

> **注意**: `last_action` 的 clip=(−100, 100) 几乎等于不裁剪。MuJoCo 和 spark_play 需注意保持一致。

### 3.4 Gait Phase 计算

```python
# 正弦步态，周期 0.6s
global_phase = (episode_length_buf * step_dt) % period / period
sin_pos = sin(2π * phase)

stance_mask[:, 0] = (sin_pos >= 0)   # 左脚 stance
stance_mask[:, 1] = (sin_pos < 0)    # 右脚 stance

# 站立时（cmd < 0.02）双脚 stance
stance_mask[cmd_norm < 0.02] = [1, 1]
```

### 3.5 动作空间 (12-dim)

```python
JointPositionActionCfg(
    joint_names=[左髋6关节, 右髋6关节],
    scale=0.25,
    use_default_offset=True,   # offset = default_joint_pos
    preserve_order=True
)
```

**关节顺序**：
```
[L_hip_pitch, L_hip_roll, L_hip_yaw, L_knee, L_ankle_pitch, L_ankle_roll,
 R_hip_pitch, R_hip_roll, R_hip_yaw, R_knee, R_ankle_pitch, R_ankle_roll]
```

### 3.6 奖励函数 (20+ 项)

#### 任务奖励

| 名称 | 权重 | 函数 | 说明 |
|------|------|------|------|
| `track_lin_vel_xy` | +1.0 | `track_lin_vel_xy_yaw_frame_exp` | 线速度跟踪（指数核，std=√0.25） |
| `track_ang_vel_z` | +0.5 | `track_ang_vel_z_exp` | 角速度跟踪（指数核） |
| `alive` | +0.15 | `is_alive` | 存活奖励 |

#### 惩罚项

| 名称 | 权重 | 说明 |
|------|------|------|
| `base_linear_velocity` | −2.0 | Z轴线速度 L2 |
| `base_angular_velocity` | −0.05 | Roll/Pitch 角速度 L2 |
| `joint_vel` | −0.001 | 关节速度 L2 |
| `joint_acc` | −2.5e−7 | 关节加速度 L2 |
| `action_rate` | −0.05 | 动作平滑性 L1 |
| `dof_pos_limits` | −5.0 | 关节限位惩罚 |
| `energy` | −2e−5 | |qvel × qfrc| 能量消耗 |
| `joint_deviation_legs` | −0.7 | 髋横滚/偏航偏离默认值 L1 |

#### 姿态/高度

| 名称 | 权重 | 说明 |
|------|------|------|
| `flat_orientation_l2` | −5.0 | 身体直立惩罚 |
| `base_height` | −10.0 | 目标高度 0.7m L2 |
| `stand_still` | −3.5 | 低命令时关节偏离默认值（cmd < 0.05） |

#### 脚部奖励

| 名称 | 权重 | 说明 |
|------|------|------|
| `feet_contact_number` | +0.5 | 步态接触匹配（与 gait_phase 一致奖励） |
| `feet_slide` | −0.2 | 脚部滑动惩罚 |
| `feet_clearance` | +1.0 | 摆动脚抬高度奖励（exp 核，target=0.1m） |
| `undesired_contacts` | −1.0 | 非脚踝接触（力 > 1N） |

### 3.7 终止条件

| 条件 | 说明 |
|------|------|
| `time_out` | 达到 20s episode 长度 |
| `base_height` | 躯干高度 < 0.2m |
| `bad_orientation` | 偏航角偏差 > 0.8 rad |

### 3.8 课程学习

| 名称 | 逻辑 |
|------|------|
| `terrain_levels` | 地形难度随训练递进 |
| `lin_vel_cmd_levels` | 当 track_lin_vel_xy reward > weight×0.8 时，扩展速度范围 ±0.1 |
| `ang_vel_cmd_levels` | 当 track_ang_vel_z reward > weight×0.8 时，扩展角速度范围 ±0.1 |

### 3.9 随机化 (EventCfg)

| 事件 | 模式 | 说明 |
|------|------|------|
| `physics_material` | startup | 摩擦随机化 (0.3, 1.0) |
| `add_base_mass` | startup | pelvis 质量缩放 ×0.7~1.3 |
| `randomize_rigid_body_mass_others` | startup | 全身质量缩放 ×0.7~1.3 |
| `base_external_force_torque` | reset | 外力/扭矩（当前 0） |
| `reset_base` | reset | 初始位置/速度随机 |
| `reset_robot_joints` | reset | 关节位置/速度随机 |
| `push_robot` | interval(5s) | 推力扰动 vel ±0.5 m/s |

---

## 4. PPO Agent 配置

**文件**: `tasks/locomotion/agents/rsl_rl_ppo_cfg.py`

### 网络结构

```
Actor:  Linear(235, 512) → ELU → Linear(512, 256) → ELU → Linear(256, 128) → ELU → Linear(128, 12)
Critic: 同结构（输入 237-dim 含 contact_mask）
```

- 激活函数: ELU
- 初始噪声标准差: 1.0

### 超参数

| 参数 | 值 |
|------|------|
| `num_steps_per_env` | 24 |
| `max_iterations` | 50000 |
| `save_interval` | 100 |
| `learning_rate` | 1e−3 (adaptive schedule) |
| `gamma` | 0.99 |
| `lam` (GAE λ) | 0.95 |
| `clip_param` | 0.2 |
| `entropy_coef` | 0.01 |
| `value_loss_coef` | 1.0 |
| `num_learning_epochs` | 5 |
| `num_mini_batches` | 4 |
| `desired_kl` | 0.01 |
| `max_grad_norm` | 1.0 |
| `empirical_normalization` | False |
| `use_clipped_value_loss` | True |

### Batch Size 计算

```
batch_size = num_envs × num_steps_per_env = 4096 × 24 = 98,304
mini_batch_size = batch_size / num_mini_batches = 24,576
```

---

## 5. 机器人参数

### 5.1 URDF

**文件**: `data/robots/magicbot-Z1/urdf/MagicBotZ1_12dof_arm_ready_pos.urdf`
（使用 arm_ready_pos 版本，手臂固定在就绪姿态）

### 5.2 关节参数

| 关节 | KP | KD | Armature | Effort Limit |
|------|------|------|----------|-------------|
| L/R hip_pitch | 100 | 4 | 0.02863 | 120 |
| L/R hip_roll | 100 | 4 | 0.02863 | 120 |
| L/R hip_yaw | 100 | 4 | 0.02863 | 120 |
| L/R knee | 150 | 5 | 0.02863 | 120 |
| L/R ankle_pitch | 60 | 3 | 0.01503 | 50 |
| L/R ankle_roll | 60 | 3 | 0.01503 | 50 |

### 5.3 默认关节位置（站立姿态）

```python
default_pos = [-0.35, 0, 0, 0.7, -0.35, 0] × 2  # 左右对称
#              pitch  roll yaw knee  pitch  roll
```

### 5.4 动作缩放

- Action scale = 0.25
- `target_pos = default_pos + action × 0.25`
- 关节速度限制: 腿部 20 rad/s，脚踝 15 rad/s

---

## 6. 训练脚本

### 6.1 train.py

**文件**: `scripts/rsl_rl/train.py`

```
python scripts/rsl_rl/train.py \
    --task Magiclab-Z1-12dof-Velocity \
    --headless \
    --num_envs 4096 \
    --device cuda:0 \
    --run_name z1_locomotion_v1 \
    --max_iterations 50000
```

**恢复训练**：
```
python scripts/rsl_rl/train.py \
    --task Magiclab-Z1-12dof-Velocity \
    --headless \
    --resume \
    --load_run <timestamp>_<run_name> \
    --checkpoint model_<N>.pt
```

**流程**：
1. AppLauncher 启动 Isaac Sim
2. `gym.make("Magiclab-Z1-12dof-Velocity")` 创建环境
3. `RslRlVecEnvWrapper` 包装
4. `OnPolicyRunner` 创建 PPO 训练器
5. 训练前自动调用 `export_deploy_cfg()` 生成 `deploy.yaml`
6. `runner.learn()` 开始训练

**启动脚本**: `train_bash.sh`（nohup 后台运行，日志输出到 `train_z1.log`）

### 6.2 play.py

**文件**: `scripts/rsl_rl/play.py`

```
python scripts/rsl_rl/play.py \
    --task Magiclab-Z1-12dof-Velocity \
    --num_envs 32 \
    --checkpoint model_<N>.pt
```

- 使用 `OnPolicyRunner` 加载 checkpoint
- 自动导出 JIT/ONNX 模型
- 支持视频录制 `--video`

### 6.3 play_keyboard.py

**文件**: `scripts/rsl_rl/play_keyboard.py`

- WASD/QE 键盘控制速度命令
- 相机跟随

### 6.4 play_z1_video.py

**文件**: `scripts/rsl_rl/play_z1_video.py`

- 绕过 Hydra，直接加载环境配置
- 优化用于视频录制
- `--video --video_length 200`

### 6.5 spark_play.py

**文件**: `scripts/spark_play.py`

- Spark (DGX) 专用，纯 JIT 推理
- 不依赖 rsl-rl，避免版本冲突
- 默认不裁剪动作（clip=None）
- `--video --num_envs 1`

---

## 7. 模型导出

### 7.1 export_jit.py

**文件**: `scripts/export_jit.py`

```
python scripts/export_jit.py \
    --checkpoint logs/.../model_5000.pt \
    --onnx
```

**功能**：
- 从 checkpoint 提取 actor 权重
- 支持 rsl-rl 3.x (`model_state_dict`) 和 5.x (`actor_state_dict`)
- 自动推断网络结构 [235, 512, 256, 128, 12]
- 导出 JIT (`policy.pt`) 和可选 ONNX (`policy.onnx`)
- 纯 PyTorch 操作，无需 Isaac Sim

**输出路径**: `{checkpoint_dir}/exported/policy.pt`

### 7.2 deploy.yaml 自动生成

**文件**: `source/.../utils/export_deploy_cfg.py`

训练开始时自动调用，生成 `params/deploy.yaml`，包含：

| 字段 | 内容 |
|------|------|
| `joint_ids_map` | SDK 关节名 → 训练关节序号映射 |
| `step_dt` | 控制步长 (0.02s) |
| `stiffness` / `damping` | PD 增益 |
| `default_joint_pos` | 默认关节位置 |
| `actions` | 动作缩放、偏移、裁剪 |
| `observations` | 观测项维度、缩放、clip |
| `commands` | 速度命令范围 |

---

## 8. 训练监控

### 8.1 train_monitor.py

**文件**: `scripts/train_monitor.py`

```
# 持续监控（与训练并行运行）
python scripts/train_monitor.py \
    --log_root logs/rsl_rl/magiclab_z1_12dof_velocity \
    --terrain gentle --poll_interval 120

# 一次性分析已完成的 run
python scripts/train_monitor.py --once \
    --run_dir logs/.../<run_dir>

# 自动导出最佳模型
python scripts/train_monitor.py --auto_export --terrain gentle
```

**5 项过拟合检测**：
1. **Reward 下降**: best reward 后持续衰退
2. **action_rate 异常**: 超过地形阈值
3. **Std 坍缩**: 策略标准差过低
4. **Value loss 异常**: 价值函数损失突增
5. **Entropy 异常**: 策略熵过低

**地形阈值配置**：
| 地形 | 预期 Best Range | Reward 天花板 | action_rate 警告 |
|------|----------------|--------------|-----------------|
| flat | 15k~30k | 49.0 | −0.8 |
| gentle | 20k~35k | 47.0 | −1.0 |
| rough | 25k~40k | 38.0 | −1.5 |

### 8.2 自动化编排 (Orchestrator)

**文件**: `scripts/automation/`

```
python -m automation.orchestrator \
    --plan training_plans/z1_5stage_plan.yaml \
    --start-from s4_rough_l1
```

**模块**：
| 文件 | 职责 |
|------|------|
| `orchestrator.py` | 主事件循环，管理阶段切换 |
| `stage_manager.py` | YAML 训练计划解析 |
| `config_swapper.py` | 阶段间环境配置切换 |
| `training_launcher.py` | 子进程启动训练 |
| `embedded_monitor.py` | 嵌入式过拟合监控 |
| `state_store.py` | 崩溃恢复状态持久化 |

**特性**：
- 自动 resume 上一阶段最佳模型
- 过拟合检测自动停止当前阶段
- 崩溃恢复（state 持久化到 JSON）
- Retry 策略：NaN 降学习率 ×0.5，OOM 减半 env 数量

---

## 9. 多阶段训练计划

**文件**: `training_plans/z1_5stage_plan.yaml`

```
S1_flat (10k) → S2_flat (25k) → S3_gentle (25k) → S4_rough_l1 (40k) → S4_full_terrain (50k) → S5_full (50k)
```

| 阶段 | 地形 | 迭代数 | Resume From |
|------|------|--------|-------------|
| `s1_flat` | flat | 10,000 | — |
| `s2_flat` | flat | 25,000 | s1_flat |
| `s3_gentle` | gentle | 25,000 | s2_flat |
| `s4_rough_l1` | rough | 40,000 | s3_gentle |
| `s4_full_terrain` | rough | 50,000 | s4_rough_l1 |
| `s5_full` | rough | 50,000 | s4_full_terrain |

每阶段通过独立 `configs/velocity_env_cfg_s{N}_*.py` 调整地形配置和奖励权重。

---

## 10. Sim2Sim 验证

### 10.1 mujoco_deploy.py

**文件**: `sim2sim/mujoco_deploy.py`

```
python sim2sim/mujoco_deploy.py \
    --mjcf ../magicbot-z1_description/mjcf/MAGICBOTZ1.xml \
    --policy logs/.../exported/policy.pt \
    --deploy_cfg logs/.../params/deploy.yaml \
    --keyboard
```

**功能**：
- 加载 JIT/ONNX 策略 + deploy.yaml
- Isaac Lab per-term 观测历史布局（`ObservationBuffer`）
- 匹配训练 PD 增益
- 键盘速度控制
- 摔倒检测与自动重置

### 10.2 mujoco_sim2sim.py

**文件**: `sim2sim/mujoco_sim2sim.py`

- 类架构（`Z1SimConfig`, `Z1RobotConfig`, `Z1ObsConfig`）
- EGL offscreen 视频录制（无需 X Server）
- 清晰的 `run_mujoco(policy, cfg)` 入口

### 10.3 关键实现细节

MuJoCo 与 Isaac Lab 一致性要求：
1. **观测历史布局**: per-term interleaved（非 frame-stacked）
2. **PD 增益**: 从 deploy.yaml 读取，匹配训练值
3. **Armature**: 与 Isaac Lab 一致
4. **关节阻尼**: MuJoCo 模型中清除默认阻尼（设为 0）
5. **步态相位**: 匹配训练的 sin-based 步态计算
6. **控制频率**: 物理 500Hz，策略 50Hz（decimation=10）

---

## 11. 真机部署

**文件**: `deploy/robot_deploy.py`

```
python deploy/robot_deploy.py \
    --policy policy.onnx \
    --deploy_cfg deploy.yaml \
    --robot_ip 192.168.54.111 \
    --suspension_test     # 悬空测试
```

**安全协议**：
1. 悬空测试模式验证策略输出
2. 初始速度命令为零
3. 逐步增加速度命令
4. Ctrl+C 紧急停止

**架构**：
- 线程安全传感器数据（IMU、关节状态）
- 策略推理 50Hz / 关节命令 500Hz
- 加载 ONNX 或 JIT 模型

---

## 12. 训练历史记录

### 12.1 版本命名规范

格式：`s{阶段}_{变体描述}`

| 旧名 | 新名 | 阶段 | 状态 | Best Model |
|------|------|------|------|-----------|
| `s1_flat` | `s2_flat` | S2 | OVERFITTING | model_3861 (r=47.33) |
| `s2_flat_retry` | `s2_flat_retry` | S2 | OVERFITTING | model_3861 (r=47.33) |
| `s2_stable` | `s2_stable` | S2 | 失败 | model_1555 (r=28.93) |
| `s3_highspeed` | `s3_highspeed` | 探索 | OVERFITTING | model_2997 (r=30.11) |
| `s4_terrain` | `s3_rough_fail` | S3 | 失败 | model_1933 (r=1.85) |
| `s4_gentle_terrain` | `s3_gentle` | S3 | HEALTHY | model_47862 (r=47.06) |
| `s5_rough_terrain` | `s4_rough_l2` | S4 | OVERFITTING | model_32790 (r=38.04) |

### 12.2 全部训练 Run（20 个）

| Run 目录 | 模型数 | 最新 Checkpoint | 备注 |
|----------|--------|-----------------|------|
| `2026-04-30_04-53-17_z1_locomotion_s1_flat` | 0 | — | 已清理 |
| `2026-04-30_14-55-05_z1_locomotion_s2_stable` | 0 | — | 已清理 |
| `2026-05-01_01-21-35_z1_locomotion_s3_highspeed` | 0 | — | 已清理 |
| `2026-05-01_01-31-15_z1_locomotion_s4_terrain` | 0 | — | 已清理 |
| `2026-05-01_04-44-07_z1_locomotion_s2_flat_retry` | 0 | — | 已清理 |
| `2026-05-01_04-50-05_z1_locomotion_s4_gentle_terrain` | **501** | model_47900 | 最佳 reward: 47.06 (HEALTHY) |
| `2026-05-01_07-04-35_z1_locomotion_s5_rough_terrain` | 0 | — | 已清理 |
| `2026-05-04_11-19-50_s6_l1_action_rate` | 18 | model_1700 | 单卡 L1 惩罚 |
| `2026-05-04_11-58-36_z1_locomotion_s7_multigpu_s4` | 0 | — | 失败 |
| `2026-05-04_11-58-40_z1_locomotion_s7_multigpu_s4` | 0 | — | 失败 |
| `2026-05-04_11-58-41_z1_locomotion_s7_multigpu_s4` | 0 | — | 失败 |
| `2026-05-04_12-04-00_z1_multigpu_test2gpu` | 0 | — | 测试 |
| `2026-05-04_12-08-02_z1_2gpu_v3` | 0 | — | 失败 |
| `2026-05-04_12-30-56_z1_mgpu_s6` | 2 | model_50018 | |
| `2026-05-04_12-34-00_z1_mgpu_4gpu_stage4` | 1 | model_50000 | |
| `2026-05-04_12-40-26_s6_l1_action_rate_4gpu` | **66** | model_8200 | 4 卡 L1 惩罚 |
| `2026-05-04_16-56-05_s4_full_terrain` | **101** | model_15000 | |
| `2026-05-05_02-03-13_s4_full_terrain` | 1 | model_15000 | |
| `2026-05-05_02-09-31_s4_full_terrain` | 3 | model_15200 | |
| `2026-05-05_02-32-45_s4_full_terrain` | **11** | model_16000 | |
| `2026-05-05_04-47-06_s5_flat_deploy` | **57** | model_5600 | 当前活跃 |

> 模型数=0 表示 checkpoint 已被清理，仅保留目录结构。加粗为有价值的 run。

---

## 13. 完整文件索引

### 核心源码

| 文件 | 路径 | 说明 |
|------|------|------|
| 环境配置 | `source/.../tasks/locomotion/robots/z1/12dof/velocity_env_cfg.py` | 场景、MDP 全部配置 |
| Agent 配置 | `source/.../tasks/locomotion/agents/rsl_rl_ppo_cfg.py` | PPO 超参数 |
| 观测函数 | `source/.../tasks/locomotion/mdp/observations.py` | gait_phase, contact_mask |
| 奖励函数 | `source/.../tasks/locomotion/mdp/rewards.py` | 自定义奖励函数 |
| 课程学习 | `source/.../tasks/locomotion/mdp/curriculums.py` | 速度递进 |
| 速度命令 | `source/.../tasks/locomotion/mdp/commands/velocity_command.py` | 带 limit_ranges |
| 机器人配置 | `source/.../assets/robots/magiclab.py` | URDF、PD 增益、关节映射 |
| 导出工具 | `source/.../utils/export_deploy_cfg.py` | 自动生成 deploy.yaml |
| 任务注册 | `source/.../tasks/locomotion/robots/z1/12dof/__init__.py` | Gym 注册 |

### 脚本

| 文件 | 路径 | 说明 |
|------|------|------|
| 训练 | `scripts/rsl_rl/train.py` | RSL-RL 训练入口 |
| 评估 | `scripts/rsl_rl/play.py` | OnPolicyRunner 评估 |
| 键盘评估 | `scripts/rsl_rl/play_keyboard.py` | 键盘控制 |
| 视频录制 | `scripts/rsl_rl/play_z1_video.py` | 绕过 Hydra 的视频录制 |
| Spark 评估 | `scripts/spark_play.py` | JIT-only，无 rsl-rl 依赖 |
| JIT 导出 | `scripts/export_jit.py` | checkpoint → JIT/ONNX |
| 训练监控 | `scripts/train_monitor.py` | 过拟合检测 |
| 编排器 | `scripts/automation/orchestrator.py` | 多阶段训练编排 |

### 部署 & Sim2Sim

| 文件 | 路径 | 说明 |
|------|------|------|
| MuJoCo 部署 | `sim2sim/mujoco_deploy.py` | Sim2Sim 验证 |
| MuJoCo 录制 | `sim2sim/mujoco_sim2sim.py` | EGL 视频录制 |
| 真机部署 | `deploy/robot_deploy.py` | 实体机器人控制 |

### Shell 脚本

| 文件 | 说明 |
|------|------|
| `train_bash.sh` | nohup 后台训练启动 |
| `play_bash.sh` | 评估启动 |
| `play_keyboard_bash.sh` | 键盘控制评估 |
| `magiclab_rl_lab.sh` | Isaac Lab 环境启动 |

---

## 14. 远程连接

- **RTX 6000D**: `phh@192.168.120.155`，VPN: iNode
- **DGX Spark**: `ssh spark`，VPN: aTrust，ARM64 架构
- 详见: [Remote_GPU_Server_Connection.md](../../docs/Remote_GPU_Server_Connection.md)

---

## 15. 常用命令速查

### RTX 服务器

```bash
# 训练状态
ssh phh@192.168.120.155 'ps aux | grep train | grep phh | grep -v grep'

# GPU 占用
ssh phh@192.168.120.155 'nvidia-smi'

# 训练日志
ssh phh@192.168.120.155 'tail -30 /tmp/z1_train_s6.log'

# 列出模型
ssh phh@192.168.120.155 'ls -lh ~/magiclab_rl_lab/logs/rsl_rl/magiclab_z1_12dof_velocity/<RUN>/model_*.pt'

# 导出 JIT
ssh phh@192.168.120.155 "cd ~/magiclab_rl_lab && source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab && python scripts/export_jit.py --checkpoint logs/.../model_N.pt"

# 过拟合检测
ssh phh@192.168.120.155 "cd ~/magiclab_rl_lab && source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab && python scripts/train_monitor.py --once --terrain gentle"

# EGL 预检（Isaac Sim 需要的 NVIDIA EGL vendor 文件）
ssh phh@192.168.120.155 'test -f ~/miniconda3/envs/isaaclab/share/glvnd/egl_vendor.d/10_nvidia.json && echo "OK" || echo "MISSING"'
```

### Spark 服务器

```bash
# 播放策略 (带画面，DISPLAY=:1 是 Spark 物理屏幕)
ssh spark "export DISPLAY=:1 && export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 && source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab && cd ~/magiclab_rl_lab && python scripts/spark_play.py --task Magiclab-Z1-12dof-Velocity --policy <path> --num_envs 1 --device=cuda:0"

# 录像
ssh spark "export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 && source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab && cd ~/magiclab_rl_lab && python scripts/spark_play.py --task Magiclab-Z1-12dof-Velocity --policy <path> --headless --video --video_length 200 --num_envs 1"
```

### 本地

```bash
# 部署到 Spark
bash RTX6000/spark_deploy.sh <RUN_DIR>

# 视频标签
python RTX6000/Magicbot_Z1/scripts/label_video.py video.mp4 --model model_N --run s4_gentle --reward 47.06

# 2×2 对比
python RTX6000/Magicbot_Z1/scripts/compare_videos.py v1.mp4 v2.mp4 v3.mp4 v4.mp4 --labels "A" "B" "C" "D"
```

### 关键配置文件速查

| 文件 | 路径 | 控制什么 |
|------|------|----------|
| 环境配置 | `source/.../z1/12dof/velocity_env_cfg.py` | 奖励权重、观测、地形、PD 参数、动作缩放 |
| PPO 配置 | `source/.../agents/rsl_rl_ppo_cfg.py` | 学习率、熵系数、GAE、网络结构 |
| 奖励函数 | `source/.../mdp/rewards.py` | 各奖励项的数学定义 |
| 观测定义 | `source/.../mdp/observations.py` | 观测向量各维度含义 |
| 训练计划 | `training_plans/z1_5stage_plan.yaml` | 阶段顺序、迭代数、配置切换 |
| 粗地形配置 | `source/.../z1/12dof/velocity_env_cfg_s7_rough_full.py` | v7 粗地形专用配置 |
