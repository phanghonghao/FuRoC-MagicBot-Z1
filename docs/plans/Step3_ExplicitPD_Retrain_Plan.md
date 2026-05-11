# Step 3 训练方案：显式 PD + 增强 Domain Randomization

## 背景

### 当前问题
s2_gentle 训练使用 `ImplicitActuator`（隐式 PD），MuJoCo 部署使用显式 PD。
sim2sim 实验表明，所有部署侧调参（Kd 增大、动作平滑、摩擦调整）均无效（14 falls/20s 不变）。

### 根因确认
```
# s2_gentle 训练时用的执行器（从 params/env.yaml 确认）：
class_type: isaaclab.actuators.actuator_pd:ImplicitActuator  ← 隐式 PD

# MuJoCo 部署时手动计算：
τ = Kp * (target - q_cur) - Kd * dq_cur                    ← 显式 PD
```

隐式 PD（PhysX 内部求解，无条件稳定）vs 显式 PD（手动计算，可能震荡）的差异
是 sim2sim gap 的根本来源，无法通过部署侧调参消除。

### 解决方案
训练时也使用显式 PD（`IdealPDActuatorCfg`），使两侧 PD 公式一致。
配合增强的 domain randomization，让策略对残余物理差异鲁棒。

---

## 修改 1：执行器配置（已完成）

**文件**: `source/magiclab_rl_lab/magiclab_rl_lab/assets/robots/magiclab.py`

当前状态：已经是 `IdealPDActuatorCfg`（s5 实验时修改）。

```python
# 当前 magiclab.py 第 106 行
actuators={
    "legs": IdealPDActuatorCfg(           # ← 已改为显式 PD
        joint_names_expr=[...],
        effort_limit=120,
        stiffness={
            ".*_hip_pitch_joint": 100.0,
            ".*_hip_roll_joint": 100.0,
            ".*_hip_yaw_joint": 100.0,
            ".*_knee_joint": 150.0,
        },
        damping={
            ".*_hip_pitch_joint": 4.0,
            ".*_hip_roll_joint": 4.0,
            ".*_hip_yaw_joint": 4.0,
            ".*_knee_joint": 5.0,
        },
        armature={
            ".*_hip.*": 0.02863,
            ".*_knee.*": 0.02863,
        },
    ),
    "feet": IdealPDActuatorCfg(           # ← 已改为显式 PD
        effort_limit=50,
        stiffness=60.0,
        damping=3.0,
        armature=0.01503,
    ),
}
```

**无需修改此文件。**

---

## 修改 2：增强 Domain Randomization

**文件**: `source/magiclab_rl_lab/magiclab_rl_lab/tasks/locomotion/robots/z1/12dof/velocity_env_cfg.py`

### 2.1 摩擦随机化范围扩大

```python
# 当前 EventCfg.physics_material：
"static_friction_range": (0.3, 1.0),
"dynamic_friction_range": (0.3, 1.0),

# 修改为（参考 Humanoid-Gym）：
"static_friction_range": (0.1, 2.0),
"dynamic_friction_range": (0.1, 2.0),
```

**原理**：更宽的摩擦范围让策略学会适应从非常滑到非常粘的地面，
覆盖 PhysX 和 MuJoCo 之间的摩擦差异。

### 2.2 质量随机化范围扩大

```python
# 当前 EventCfg.add_base_mass：
"mass_distribution_params": (0.7, 1.3),

# 修改为：
"mass_distribution_params": (0.5, 1.5),
```

**原理**：更宽的质量范围增强策略的鲁棒性，
适应不同物理引擎的质量积分差异。

### 2.3 推力扰动增强

```python
# 当前 EventCfg.push_robot：
"interval_range_s": (5.0, 5.0),
"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}

# 修改为：
"interval_range_s": (3.0, 5.0),       # 更频繁
"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}  # 更强
```

**原理**：更频繁更强的扰动让策略学会从失衡中恢复，
减少对特定物理引擎稳定性的依赖。

---

## 修改 3：观测噪声增强

**文件**: `velocity_env_cfg.py` 的 `ObservationsCfg.PolicyCfg`

```python
# 当前噪声设置：
base_ang_vel  = ObsTerm(..., noise=Unoise(n_min=-0.2, n_max=0.2))
projected_gravity = ObsTerm(..., noise=Unoise(n_min=-0.1, n_max=0.1))
joint_pos_rel = ObsTerm(..., noise=Unoise(n_min=-0.02, n_max=0.02))
joint_vel_rel = ObsTerm(..., noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.05)

# 建议增大（参考 Humanoid-Gym）：
base_ang_vel  = ObsTerm(..., noise=Unoise(n_min=-0.3, n_max=0.3))    # ±0.2 → ±0.3
projected_gravity = ObsTerm(..., noise=Unoise(n_min=-0.15, n_max=0.15))  # ±0.1 → ±0.15
joint_pos_rel = ObsTerm(..., noise=Unoise(n_min=-0.05, n_max=0.05))   # ±0.02 → ±0.05
joint_vel_rel = ObsTerm(..., noise=Unoise(n_min=-2.0, n_max=2.0), scale=0.05)  # ±1.5 → ±2.0
```

**原理**：更大的观测噪声让策略不依赖精确的传感器读数，
适应不同物理引擎的观测偏差。

---

## 修改 4：Reward 调整

**文件**: `velocity_env_cfg.py` 的 `RewardsCfg`

显式 PD 比隐式 PD 更容易震荡，需要调整 reward 鼓励平滑动作：

```python
# 增大 action_rate 惩罚（鼓励平滑动作）
action_rate = RewTerm(func=mdp.action_rate_l1, weight=-0.1)  # 从 -0.05 增大到 -0.1

# 可选：增大 joint_acc 惩罚（抑制关节加速度过大）
joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-5e-7)  # 从 -2.5e-7 增大
```

---

## 完整的 velocity_env_cfg.py 修改对比

### EventCfg 部分

| 参数 | 旧值 (s2_gentle) | 新值 (s6_explicit_pd) |
|------|-----------------|---------------------|
| friction_range | (0.3, 1.0) | **(0.1, 2.0)** |
| mass_range (pelvis) | (0.7, 1.3) | **(0.5, 1.5)** |
| mass_range (others) | (0.7, 1.3) | **(0.5, 1.5)** |
| push interval | (5.0, 5.0) | **(3.0, 5.0)** |
| push velocity | ±0.5 | **±1.0** |

### ObservationsCfg.PolicyCfg 部分

| 观测项 | 旧噪声 | 新噪声 |
|--------|--------|--------|
| base_ang_vel | ±0.2 | **±0.3** |
| projected_gravity | ±0.1 | **±0.15** |
| joint_pos_rel | ±0.02 | **±0.05** |
| joint_vel_rel | ±1.5 | **±2.0** |

### RewardsCfg 部分

| Reward | 旧权重 | 新权重 |
|--------|--------|--------|
| action_rate | -0.05 | **-0.1** |
| joint_acc | -2.5e-7 | **-5e-7** |

---

## 启动方式

### 通过 --automation（5-phase pipeline）

```bash
# 1. 先更新 velocity_env_cfg.py（上面列出的修改）
# 2. 确认 magiclab.py 已是 IdealPDActuatorCfg
# 3. 启动自动化训练

/gpu-train --start --from s2_gentle --run s6_explicit_pd_dr --gpus 4
```

### 或手动启动（单次训练）

```bash
ssh phh@192.168.120.155 "cd ~/magiclab_rl_lab && \
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab && \
  nohup python -u scripts/rsl_rl/train.py \
    --task Magiclab-Z1-12dof-Velocity \
    --run_name s6_explicit_pd_dr \
    --headless --num_envs 16384 --device cuda:0 \
    --max_iterations 50000 \
  > /tmp/z1_train_s6.log 2>&1 & echo PID=\$!"
```

---

## 验证计划

### 训练中验证
1. 启动后检查 `params/env.yaml`，确认 `class_type: IdealPDActuator`（不是 ImplicitActuator）
2. ~500 iter 后检查 reward 趋势（应为正值且上升）
3. ~2000 iter 后录制 Isaac Lab 视频，确认步态正常

### 训练后 sim2sim 验证
1. 导出 JIT policy：`python scripts/export_jit.py --checkpoint <best_model>`
2. 用 `mujoco_humanoid_gym.py`（原始参数，无 Kd boost）测试
3. 对比 s2_gentle（14 falls/20s）vs s6_explicit_pd 的摔倒次数
4. 目标：摔倒次数 < 5 次/20s，或平均站立时间 > 5s

### 关键对比指标

| 指标 | s2_gentle (implicit PD) | s6_explicit_pd (目标) |
|------|------------------------|---------------------|
| MuJoCo 摔倒/20s | 14 | **< 5** |
| 平均站立时间 | ~1.4s | **> 5s** |
| 需要手动 Kd boost | 是 | **否** |

---

## 参考资料

- Isaac Lab 执行器对比：`IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py`
  - `ImplicitActuator`：设 position/velocity target，PhysX 内部 PD
  - `IdealPDActuator`：手动计算 τ = Kp*error + Kd*error_vel，直接设 effort
- Humanoid-Gym 做法：两侧都用显式 PD + 摩擦随机化 0.1-2.0 + 质量随机化 ±5kg
- s5_explicit_pd_16k 实验记录：用 IdealPDActuatorCfg 训练，但只跑到 iter 1087（overfitting），
  说明显式 PD 本身可以训练，但需要更完整的训练流程和更好的超参
