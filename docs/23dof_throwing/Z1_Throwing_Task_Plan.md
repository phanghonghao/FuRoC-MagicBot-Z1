# Z1 投篮 RL 任务 — 完整实施计划

> 任务类型：23DOF 投篮（非行走，独立任务体系）
> 目标：在 IsaacLab 中用 23DOF URDF 训练「球预置在手掌上 → 投篮」的 RL 策略
> 日期：2026-05-11

---

## 1. 设计概览

### 1.1 问题定义

- **机器人**：Z1 人形，23DOF（腿12 + 手臂10 + 腰1）
- **手部**：固定手掌（无手指 DOF），只能「托球 + 投掷」
- **任务**：球预置在手掌上 → 手臂挥动 → 将球投入目标篮筐
- **手部可替换设计**：当前用简单 sphere collision，未来替换 Z1 灵巧手 URDF 时只需改配置

### 1.2 关键设计决策

| 决策 | 方案 |
|------|------|
| 手部 collision | URDF 中用 sphere，预留 TODO 标记 |
| 手部参数化 | `HAND_CONFIG` 字典，控制动作空间/观测维度 |
| 球的表示 | `RigidObject`，reset 时预置在手掌 link 上 |
| 篮筐 | 静态 `AssetBaseCfg`（圆柱 + 网可视化） |
| 奖励函数 | 分阶段：保持球 → 挥臂方向 → 释放后轨迹 → 进球 |
| 动作空间 | 手臂 10DOF + 腰1DOF（当前），灵巧手时扩展 |

---

## 2. 文件结构

```
magiclab_rl_lab/
├── assets/robots/magiclab.py            # + MAGICLAB_Z1_23DOF_CFG（新增）
├── data/robots/magicbot-Z1/urdf/
│   └── MagicBotZ1_23dof.urdf           # 手掌加 collision（已修改）
└── tasks/
    ├── locomotion/                      # 原有 12DOF 行走任务（不动）
    └── throwing/                        # 新任务目录
        ├── __init__.py                  # from .robots import *
        ├── agents/
        │   ├── __init__.py
        │   └── rsl_rl_ppo_cfg.py       # PPO 超参
        ├── mdp/
        │   ├── __init__.py              # re-export isaaclab mdp + 自定义
        │   ├── rewards.py               # 投篮奖励函数
        │   ├── observations.py          # 投篮观测函数
        │   ├── commands.py              # 目标位置 command
        │   └── curriculums.py           # 距离 curriculum
        └── robots/
            ├── __init__.py              # import_packages
            └── z1/
                ├── __init__.py          # gym.register("Magiclab-Z1-23dof-Throwing")
                └── shoot_env_cfg.py     # 环境配置（含 HAND_CONFIG）
```

---

## 3. Step-by-Step 实施步骤

### Step 1: 修改 URDF — 手掌 collision

**文件**：`data/robots/magicbot-Z1/urdf/MagicBotZ1_23dof.urdf`

在 `left_hand_palm_link` 和 `right_hand_palm_link` 的 `<inertial>` 后加入 collision：

```xml
<!-- TODO: 替换为 Z1 真实手掌 URDF（预计几天后到） -->
<collision>
  <origin xyz="0.05 0 0" rpy="0 0 0"/>
  <geometry>
    <sphere radius="0.04"/>
  </geometry>
</collision>
```

### Step 2: 添加 23DOF 机器人配置

**文件**：`assets/robots/magiclab.py`

新增 `MAGICLAB_Z1_23DOF_CFG`：
- URDF: `MagicBotZ1_23dof.urdf`
- 6 组 actuator：legs / feet / shoulders / elbows / wrists / waist
- 初始站姿（腿同 12DOF）+ 手臂初始角度（肩膀前伸 0.3rad，肘部 -0.5rad）
- 23 个 SDK 关节名

### Step 3: 创建投篮 mdp 模块

#### 3.1 投篮奖励函数 (`mdp/rewards.py`)

| 函数 | 类型 | 描述 |
|------|------|------|
| `ball_on_palm` | 正奖励 | 球保持在手掌上（距离 < 阈值） |
| `ball_release_velocity_reward` | 正奖励 | 释放时球速度方向朝向篮筐 |
| `ball_distance_to_target` | 正奖励 | 球距篮筐越近越好 |
| `ball_in_basket` | 正奖励 | 球进入篮筐（稀疏，大奖励） |
| `ball_fall_penalty` | 负奖励 | 球提前掉落 |
| `arm_energy_penalty` | 负奖励 | 手臂能耗惩罚 |
| `joint_limit_penalty` | 负奖励 | 关节限位惩罚 |

#### 3.2 投篮观测函数 (`mdp/observations.py`)

| 函数 | 描述 |
|------|------|
| `ee_position` | 末端执行器（手掌）位置（base frame） |
| `ee_velocity` | 末端执行器速度（base frame） |
| `ball_position` | 球的位置（base frame） |
| `ball_velocity` | 球的速度（base frame） |
| `ball_on_palm_flag` | 球是否在手掌上（0/1） |
| `target_position` | 篮筐目标位置（base frame） |

#### 3.3 目标命令 (`mdp/commands.py`)

`UniformTargetCommandCfg`：随机采样篮筐位置（距离 + 方位角 + 高度）

#### 3.4 课程学习 (`mdp/curriculums.py`)

`target_distance_levels`：逐步增加投篮距离

### Step 4: 创建环境配置

**文件**：`tasks/throwing/robots/z1/shoot_env_cfg.py`

核心配置：
- `HAND_CONFIG`：手部参数化变量（换 URDF 只改这里）
- `ThrowingSceneCfg`：地面 + 机器人 + 球（RigidObject）+ 篮筐 + 灯光
- `ActionsCfg`：手臂 10DOF + 腰 1DOF = 11 个动作关节
- `ObservationsCfg`：Policy（本体态 + 球状态 + 目标 + EE）+ Critic（+ base_lin_vel 等）
- `RewardsCfg`：9 个奖励项（4 正 + 5 负）
- `TerminationsCfg`：超时 / 基座高度过低 / 姿态异常

### Step 5: 注册环境

**文件**：`tasks/throwing/robots/z1/__init__.py`

```python
gym.register(
    id="Magiclab-Z1-23dof-Throwing",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shoot_env_cfg:ThrowingEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.shoot_env_cfg:ThrowingPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"magiclab_rl_lab.tasks.throwing.agents.rsl_rl_ppo_cfg:ThrowingPPORunnerCfg",
    },
)
```

### Step 6: 训练命令

```bash
# 在 RTX 服务器上执行
cd ~/magiclab_rl_lab
conda activate isaaclab

# 单 GPU 训练
python -m isaaclab.app.runner \
    --task Magiclab-Z1-23dof-Throwing \
    --num_envs 4096 \
    --headless

# 多 GPU 训练
torchrun --nproc_per_node=2 -m isaaclab.app.runner \
    --task Magiclab-Z1-23dof-Throwing \
    --num_envs 8192 \
    --headless
```

### Step 7: 验证

1. 环境能正常启动，球预置在手掌上
2. 机器人不会立即摔倒
3. 球不会立即掉落（保持球奖励生效）
4. 训练 1000 iter 后机器人开始尝试挥臂
5. 训练 10000 iter 后能看到一些投掷行为
6. 最终成功率 > 60%

---

## 4. 奖励函数设计详情

### 4.1 分阶段奖励（Phase-based）

```
阶段1：保持球 (Phase: 0.0 - 0.3)
├── ball_on_palm:         weight=2.0   # 球在手掌上
├── robot_balance:        weight=-1.0  # 不要摔倒
└── arm_energy_penalty:   weight=-0.01 # 不要过度发力

阶段2：准备挥臂 (Phase: 0.3 - 0.6)
├── ball_on_palm:         weight=1.0   # 球仍然在手掌上
├── ee_backswing:         weight=0.5   # 手臂后摆准备
└── arm_energy_penalty:   weight=-0.01

阶段3：投掷 (Phase: 0.6 - 1.0)
├── ball_release_vel:     weight=3.0   # 释放速度方向
├── ball_trajectory:      weight=2.0   # 轨迹评分
├── ball_in_basket:       weight=10.0  # 进球（稀疏大奖励）
└── ball_dist_to_basket:  weight=1.0   # 距离越近越好
```

### 4.2 通用惩罚（全程生效）

```
├── action_rate:          weight=-0.01  # 动作平滑
├── joint_limit_penalty:  weight=-1.0   # 关节限位
├── base_height_penalty:  weight=-0.5   # 保持站立
├── ball_fall_penalty:    weight=-2.0   # 球提前掉落
```

---

## 5. 手部参数化设计

### 5.1 HAND_CONFIG（换 URDF 只改这里）

```python
HAND_CONFIG = {
    "hand_type": "fixed_palm",         # 或 "dexterous"（未来切换）
    "hand_joint_names": [],            # 灵巧手时填入手指关节名
    "hand_dof": 0,                     # 灵巧手 DOF 数
    "ee_link": "left_hand_palm_link",  # 末端执行器 link
    "ball_support_link": "left_hand_palm_link",
}
```

### 5.2 动作空间动态生成

```python
if HAND_CONFIG["hand_type"] == "fixed_palm":
    ACTION_JOINTS = ARM_JOINTS + WAIST_JOINTS   # 11 DOF
elif HAND_CONFIG["hand_type"] == "dexterous":
    ACTION_JOINTS = ARM_JOINTS + HAND_CONFIG["hand_joint_names"] + WAIST_JOINTS
```

### 5.3 换手流程

1. **替换 URDF**：新灵巧手 URDF 替换 `MagicBotZ1_23dof.urdf` 中的手掌部分
2. **修改配置**：`shoot_env_cfg.py` 中 `HAND_CONFIG["hand_type"] = "dexterous"` + 填入关节名
3. **训练逻辑不变**：动作空间自动扩展，奖励/观测代码无需修改

---

## 6. PPO 超参

| 参数 | 值 |
|------|-----|
| num_steps_per_env | 48 |
| max_iterations | 100000 |
| save_interval | 200 |
| actor_hidden_dims | [256, 128, 64] |
| critic_hidden_dims | [256, 128, 64] |
| learning_rate | 5e-4 |
| entropy_coef | 0.02 |
| num_learning_epochs | 8 |
| num_mini_batches | 4 |

---

## 7. 环境参数

| 参数 | 值 |
|------|-----|
| num_envs | 4096 |
| env_spacing | 5.0m |
| episode_length | 5.0s |
| dt | 0.004s |
| decimation | 8 |
| 球半径 | 0.06m |
| 球质量 | 0.15kg |
| 篮筐半径 | 0.15m |

---

## 8. 注意事项

- 球的物理参数（摩擦、弹性）需要调参，初始建议 friction=0.8, restitution=0.7
- 篮筐碰撞检测用 trigger（不需要物理反弹）
- 球预置位置需要在 `reset` 事件中精确设置到手掌 link 上方
- 建议先在近距离（1m）训练成功后，再用 curriculum 扩展到更远距离
- 此任务与 12DOF 行走任务完全独立，互不影响
