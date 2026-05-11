# MagicBot Z1 Locomotion 训练规划

> 从最基础到最困难的分阶段训练策略、参数配置与预期训练量。
>
> 硬件：RTX 6000D (85GB)，16384 envs，单次训练约 28-35 小时 / 50K 迭代。

---

## 1. 机器人参数

### 1.1 关节布局（12DOF）

```
左腿: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
右腿: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
```

### 1.2 PD 增益与默认姿态

| 关节 | KP | KD | 默认位置 (rad) | Armature | 力矩限制 (N·m) |
|------|-----|-----|----------------|----------|----------------|
| hip_pitch | 100 | 4 | -0.35 | 0.02863 | 120 |
| hip_roll | 100 | 4 | 0 | 0.02863 | 120 |
| hip_yaw | 100 | 4 | 0 | 0.02863 | 120 |
| knee | 150 | 5 | 0.7 | 0.02863 | 120 |
| ankle_pitch | 60 | 3 | -0.35 | 0.01503 | 50 |
| ankle_roll | 60 | 3 | 0 | 0.01503 | 50 |

左右对称，初始高度 0.69m，action scale=0.25，步态周期 0.6s。

### 1.3 仿真参数

| 参数 | 值 |
|------|-----|
| 物理步长 | 0.002s (500Hz) |
| 控制频率 | 50Hz (decimation=10) |
| 回合时长 | 20s (1000步) |
| Actor 输入 | 47维 × 5帧 = 235维 |
| Actor 输出 | 12维 (关节位置增量) |
| 网络结构 | [512, 256, 128] + ELU |

---

## 2. 训练阶段规划

### 阶段总览

```
Stage 1: 站立 ──→ Stage 2: 平地行走 ──→ Stage 3: 轻度地形 ──→ Stage 4: 粗糙地形 ──→ Stage 5: 复杂地形
  (10K iter)        (20K iter)            (20K iter)            (30K iter)           (30K+ iter)
```

每个阶段在前一阶段的最佳 checkpoint 上继续训练（调整奖励和地形），不需要从零开始。

---

### Stage 1: 原地站立

**目标**：机器人学会保持站立姿态，不摔倒，跟踪零速指令。

**预期训练量**：5,000 ~ 10,000 迭代（~2-4 小时）

**关键配置变更**（相对于基础配置）：

| 参数 | 值 | 说明 |
|------|-----|------|
| 地形 | 100% flat | 平地 |
| 速度指令范围 | vx: [-0.1, 0.1] | 几乎不动 |
| `stand_still` 权重 | -5.0 (↑) | 强化站立惩罚 |
| `base_height` 权重 | -15.0 (↑) | 更强的高度保持 |
| `flat_orientation` 权重 | -8.0 (↑) | 更强的姿态保持 |
| `feet_clearance` 权重 | 0.0 (关闭) | 站立不需要抬脚 |
| `feet_contact_number` 权重 | 0.0 (关闭) | 站立不需要步态 |
| `track_lin_vel_xy` 权重 | 0.5 (↓) | 速度跟踪不重要 |
| max_iterations | 10,000 | |

**成功标准**：
- time_out > 95%
- 机器人在推力扰动下恢复平衡
- 基座高度稳定在 0.69m ± 0.02m

---

### Stage 2: 平地行走

**目标**：在平地上学会跟踪速度指令，前进/后退/转弯，形成自然步态。

**预期训练量**：15,000 ~ 25,000 迭代（~10-17 小时）

**关键配置**：

| 参数 | 值 | 说明 |
|------|-----|------|
| 地形 | 100% flat | 平地 |
| 速度指令 | vx: [-0.5, 1.0], vy: [-0.5, 0.5], vyaw: [-0.5, 0.5] | 全范围速度 |
| `track_lin_vel_xy` 权重 | 1.0 | 速度跟踪核心 |
| `track_ang_vel_z` 权重 | 0.5 | 转弯跟踪 |
| `feet_clearance` 权重 | 1.0 | 抬脚奖励 |
| `feet_contact_number` 权重 | 0.5 | 正确着地 |
| `stand_still` 权重 | -3.5 | 正常站立惩罚 |
| `base_height` 权重 | -10.0 | 标准高度保持 |
| `action_rate` 权重 | -0.05 | 动作平滑度 |
| `energy` 权重 | -2e-5 | 能效 |
| max_iterations | 25,000 | |

**课程学习**：速度指令从 [-0.1, 0.1] 逐步扩展到 [-0.5, 1.0]，由 `lin_vel_cmd_levels` 自动控制。

**成功标准**：
- time_out > 90%
- 速度跟踪误差 < 0.2 m/s
- 步态自然、左右交替、无明显跛行
- 前进速度可达 0.8 m/s

---

### Stage 3: 轻度地形（Gentle Terrain）

**目标**：适应轻微起伏的地面，能在随机网格地形上行走。

**预期训练量**：15,000 ~ 25,000 迭代（~10-17 小时）

**起始**：从 Stage 2 最佳 checkpoint 继续。

**关键配置变更**：

| 参数 | 值 | 说明 |
|------|-----|------|
| 地形 | 50% flat + 50% random_grid | 混合地形 |
| `difficulty_range` | (0.0, 0.5) | 限制最大难度 |
| `base_height` 权重 | -8.0 (↓) | 放宽高度约束（地形起伏） |
| `flat_orientation` 权重 | -3.0 (↓) | 允许轻微倾斜 |
| `feet_clearance` 权重 | 1.5 (↑) | 更强调抬脚（避免绊倒） |
| 域随机化 - 摩擦 | (0.3, 1.5) (↑) | 更大摩擦范围 |
| 域随机化 - 推力 | ±1.0 m/s (↑) | 更强扰动 |
| max_iterations | 20,000 | |

**成功标准**：
- 在 difficulty 0.3 的地形上 time_out > 80%
- 能跨过 3-5cm 高的障碍
- 不因地面起伏而摔倒

---

### Stage 4: 粗糙地形（Rough Terrain）

**目标**：在高度复杂的地形上行走，包括楼梯、沟壑、方块障碍。

**预期训练量**：25,000 ~ 40,000 迭代（~17-28 小时）

**起始**：从 Stage 3 最佳 checkpoint 继续。

**关键配置变更**：

| 参数 | 值 | 说明 |
|------|-----|------|
| 地形 | 20% flat + 30% random_grid + 20% stairs + 15% gap + 15% boxes | 全类型混合 |
| `difficulty_range` | (0.0, 1.0) | 全难度 |
| `base_height` 权重 | -5.0 (↓) | 进一步放宽 |
| `flat_orientation` 权重 | -2.0 (↓) | 允许更大倾斜 |
| `feet_clearance` 权重 | 2.0 (↑) | 强调高抬脚 |
| `undesired_contacts` 权重 | -2.0 (↑) | 避免膝盖着地 |
| 域随机化 - 质量 | (0.5, 1.5) (↑) | 更大质量变化 |
| 域随机化 - 推力 | ±1.5 m/s (↑) | 强扰动 |
| max_iterations | 40,000 | |

**成功标准**：
- 在 difficulty 0.6 的地形上 time_out > 70%
- 能上 10cm 台阶
- 能跨过 15cm 沟壑
- 推力扰动后恢复

---

### Stage 5: 复杂地形 + 高速

**目标**：全地形全速度，接近 Sim-to-Real 部署级别。

**预期训练量**：30,000 ~ 50,000 迭代（~20-35 小时）

**起始**：从 Stage 4 最佳 checkpoint 继续。

**关键配置变更**：

| 参数 | 值 | 说明 |
|------|-----|------|
| 地形 | 全类型全难度 | 含 rails（台阶） |
| 速度上限 | vx: [-1.0, 1.5] | 更高速 |
| `feet_clearance` target | 0.12m (↑) | 更高抬脚 |
| `energy` 权重 | -5e-5 (↑) | 控制能耗 |
| `action_rate` 权重 | -0.1 (↑) | 更严平滑约束 |
| 观测噪声 | 加倍 | 更鲁棒 |
| 域随机化 | 全面加强 | 质量、摩擦、推力、延迟 |
| max_iterations | 50,000 | |

**成功标准**：
- 在 difficulty 0.8+ 地形上 time_out > 60%
- 前进速度可达 1.2 m/s
- 模拟真机传感器噪声下仍稳定

---

## 3. PPO 超参数

### 3.1 基础配置（所有阶段通用）

| 参数 | 值 | 说明 |
|------|-----|------|
| learning_rate | 1e-3 | 自适应调整 (schedule=adaptive) |
| gamma | 0.99 | 折扣因子 |
| lam | 0.95 | GAE λ |
| clip_param | 0.2 | PPO clip |
| entropy_coef | 0.01 | 探索系数 |
| desired_kl | 0.01 | 目标 KL 散度 |
| num_steps_per_env | 24 | 每次采集步数 |
| num_mini_batches | 4 | Mini-batch 数 |
| num_learning_epochs | 5 | 数据重用次数 |
| max_grad_norm | 1.0 | 梯度裁剪 |
| init_noise_std | 1.0 | 初始策略噪声 |
| save_interval | 100 | 每 100 次保存 |
| empirical_normalization | False | 不做回报归一化 |

### 3.2 调参原则

| 原则 | 说明 |
|------|------|
| 每次只改 1 个参数 | 同时改多个会导致崩溃（s2_stable 教训） |
| 不要改网络结构 | [512,256,128] 已验证有效 |
| 不要盲目增大 num_envs | 16384 已用 ~15GB，需关注 VRAM |
| entropy_coef 谨慎调 | 过高→不收敛，过低→过早收敛 |
| learning_rate 保持 1e-3 | 配合 adaptive schedule 自动调节 |

### 3.3 各阶段建议调整

| 参数 | Stage 1-2 | Stage 3 | Stage 4-5 |
|------|-----------|---------|-----------|
| learning_rate | 1e-3 | 5e-4 (可降) | 5e-4 ~ 1e-3 |
| entropy_coef | 0.01 | 0.01 | 0.008 (可降) |
| action_rate 权重 | -0.05 | -0.08 | -0.1 |

---

## 4. 奖励函数参考

### 4.1 权重速查表

| 奖励项 | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 |
|--------|---------|---------|---------|---------|---------|
| track_lin_vel_xy | 0.5 | 1.0 | 1.0 | 1.0 | 1.0 |
| track_ang_vel_z | 0.3 | 0.5 | 0.5 | 0.5 | 0.5 |
| alive | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 |
| base_height | -15.0 | -10.0 | -8.0 | -5.0 | -5.0 |
| flat_orientation | -8.0 | -5.0 | -3.0 | -2.0 | -2.0 |
| dof_pos_limits | -5.0 | -5.0 | -5.0 | -5.0 | -5.0 |
| stand_still | -5.0 | -3.5 | -3.5 | -3.5 | -3.5 |
| base_lin_vel_z | -2.0 | -2.0 | -2.0 | -2.0 | -2.0 |
| joint_deviation_legs | -0.7 | -0.7 | -0.7 | -0.7 | -0.7 |
| action_rate | -0.05 | -0.05 | -0.08 | -0.1 | -0.1 |
| feet_slide | -0.2 | -0.2 | -0.3 | -0.5 | -0.5 |
| undesired_contacts | -1.0 | -1.0 | -1.5 | -2.0 | -2.0 |
| feet_clearance | 0.0 | 1.0 | 1.5 | 2.0 | 2.0 |
| feet_contact_number | 0.0 | 0.5 | 0.5 | 0.5 | 0.5 |
| energy | -2e-5 | -2e-5 | -2e-5 | -5e-5 | -5e-5 |
| joint_vel | -0.001 | -0.001 | -0.001 | -0.001 | -0.001 |
| joint_acc | -2.5e-7 | -2.5e-7 | -2.5e-7 | -2.5e-7 | -2.5e-7 |

### 4.2 调整逻辑

- **base_height**：随地形复杂度递减。平地要求精确高度，复杂地形允许浮动
- **flat_orientation**：同上，复杂地形需要身体倾斜来保持平衡
- **feet_clearance**：随地形复杂度递增。复杂地形必须高抬脚避免绊倒
- **action_rate**：随阶段递增。后期要求更平滑的步态
- **Stage 1 关闭步态奖励**：站立阶段不需要步态相关奖励

---

## 5. 地形配置

### 5.1 地形类型

| 类型 | 配置类 | 特点 | 适用阶段 |
|------|--------|------|---------|
| 平地 | `MeshPlaneTerrainCfg` | 完全平坦 | 1-2 |
| 随机网格 | `MeshRandomGridTerrainCfg` | 随机高度起伏 | 3-4 |
| 金字塔楼梯 | `MeshPyramidStairsTerrainCfg` | 上下楼梯 | 4-5 |
| 沟壑 | `MeshGapTerrainCfg` | 间隙 | 4-5 |
| 方块障碍 | `MeshRepeatedBoxesTerrainCfg` | 散落方块 | 4-5 |
| 台阶 | `MeshRailsTerrainCfg` | 横向台阶 | 5 |

### 5.2 各阶段地形配比

```python
# Stage 1-2: 纯平地
sub_terrains = {"flat": MeshPlaneTerrainCfg(proportion=1.0)}

# Stage 3: 轻度混合
sub_terrains = {
    "flat": MeshPlaneTerrainCfg(proportion=0.5),
    "random_grid": MeshRandomGridTerrainCfg(proportion=0.5),
}

# Stage 4: 全类型混合
sub_terrains = {
    "flat": MeshPlaneTerrainCfg(proportion=0.2),
    "random_grid": MeshRandomGridTerrainCfg(proportion=0.3),
    "pyramid_stairs": MeshPyramidStairsTerrainCfg(proportion=0.2),
    "gap": MeshGapTerrainCfg(proportion=0.15),
    "boxes": MeshRepeatedBoxesTerrainCfg(proportion=0.15),
}

# Stage 5: 含高难度台阶
sub_terrains = {
    "flat": MeshPlaneTerrainCfg(proportion=0.1),
    "random_grid": MeshRandomGridTerrainCfg(proportion=0.2),
    "pyramid_stairs": MeshPyramidStairsTerrainCfg(proportion=0.2),
    "gap": MeshGapTerrainCfg(proportion=0.15),
    "boxes": MeshRepeatedBoxesTerrainCfg(proportion=0.15),
    "rails": MeshRailsTerrainCfg(proportion=0.2),
}
```

### 5.3 地形生成参数

| 参数 | 值 | 说明 |
|------|-----|------|
| size | (8.0, 8.0) | 每块地形 8m×8m |
| num_rows | 9 | 9 行难度等级（课程学习） |
| num_cols | 21 | 21 列环境宽度 |
| border_width | 20.0 | 外围安全区 |
| horizontal_scale | 0.1 | 水平分辨率 0.1m |
| vertical_scale | 0.005 | 垂直分辨率 5mm |

---

## 6. 域随机化

### 6.1 各阶段随机化强度

| 随机化项 | Stage 1-2 | Stage 3 | Stage 4-5 |
|---------|-----------|---------|-----------|
| 静摩擦 | (0.3, 1.0) | (0.3, 1.5) | (0.2, 2.0) |
| 动摩擦 | (0.3, 1.0) | (0.3, 1.5) | (0.2, 2.0) |
| 骨盆质量 | (0.7, 1.3) | (0.6, 1.4) | (0.5, 1.5) |
| 全身质量 | (0.7, 1.3) | (0.6, 1.4) | (0.5, 1.5) |
| 推力 vx/vy | ±0.5 m/s | ±1.0 m/s | ±1.5 m/s |
| 初始位置 | ±0.5m | ±0.5m | ±0.5m |
| 初始速度 | ±0.5 | ±0.5 | ±0.5 |

---

## 7. 训练监控指标

### 7.1 关键指标与健康阈值

| 指标 | 健康 | 警告 | 危险 |
|------|------|------|------|
| time_out (%) | > 90% | < 80% | < 50% |
| bad_orientation (%) | < 5% | > 20% | > 50% |
| episode_length (步) | > 900 | < 500 | < 100 |
| mean_reward | > 30 | < 10 | < 0 |
| action_rate | -0.2 ~ -0.5 | > -1.0 | > -5.0 |
| value_loss | 0.01 ~ 0.1 | > 1.0 | > 100 |

### 7.2 各阶段预期指标

| 指标 | Stage 1 (10K) | Stage 2 (25K) | Stage 3 (20K) | Stage 4 (40K) |
|------|---------------|---------------|---------------|---------------|
| time_out | 98% | 92% | 80% | 70% |
| mean_reward | 40+ | 35+ | 25+ | 20+ |
| episode_length | 1000 | 950+ | 800+ | 700+ |

---

## 8. 已知问题与规避策略

### 8.1 action_rate 正反馈崩塌

**现象**：训练 25K-35K 时 action_rate 突然飙升，value_loss 爆炸，训练崩塌。

**已验证**：s4_gentle_terrain model_49999 的 raw actions 均值 ~109（正常应 < 5），策略发散。

**规避策略**：
- 使用 L1 (`action_rate_l1`) 而非 L2 — 线性增长不会爆炸
- 训练早期用较小权重 (-0.05)，后期逐步增大
- 监控 value_loss，超过 1.0 时考虑降低学习率或停止

### 8.2 超参批量修改崩溃

**已验证**：s2_stable 同时改 6 个超参 → 100 迭代内崩溃。每次只改 1 个参数。

### 8.3 地形难度过早引入

**已验证**：s4_terrain（高难度地形）在机器人还没学会走路时就引入 → 100% 失败。必须先平地训练。

### 8.4 Checkpoint 选择

- 不一定最后的模型最好（model_49999 发散，model_3700 更稳定）
- 建议每 500 迭代评估一次，选 reward 最高且 action_rate 正常的 checkpoint
- 最佳 checkpoint 窗口通常在 reward 曲线峰值附近

---

## 9. 历史训练记录

> 更新时间：2026-05-06。通过 `/gpu-train` 获取最新状态。

### 9.1 训练链

```
s1_flat (m3861, 47.33, flat)
  └──→ s2_gentle (m47862, 47.06, gentle) ← 历史最佳
         └──→ s3_rough_l2 (m32790, 38.04, rough) ← 粗地形最佳
                └──→ s4_full_terrain (m5155→m15000, 37.73, full terrain) ← 已停止
```

### 9.2 各 Run 详情

| # | Run | 迭代 | 最佳模型 | Best Reward | 状态 | 教训 |
|---|-----|------|---------|-------------|------|------|
| 1 | s1_flat | 36,429 | m3861 | 47.33 | OVERFITTING | 后续阶段 resume 起点；原始 checkpoint 已清理，仅保留 JIT |
| 2 | s2_gentle | 49,999 | **m47862** | **47.06** | HEALTHY | 历史最佳；L2 action_rate 尾部崩塌（model_49999 raw actions ~109） |
| 3 | s3_rough_l2 | 41,590 | m32790 | 38.04 | OVERFITTING | action_rate 崩塌到 -167；原始 checkpoint 已清理，仅保留 JIT |
| 4 | s4_full_terrain | 15,000 | m5155 | 37.73 | 已停止 | 4-GPU orchestrator，全类型地形 |

### 9.3 仿真视频记录

| Run | 模型 | Isaac Sim 视频 | MuJoCo 视频 |
|-----|------|---------------|-------------|
| s1_flat | m3861 (JIT) | s1_flat_model3861_isaaclab.mp4 | s1_flat_m3861_sim2sim_mujoco.mp4 |
| s2_gentle | m47900 | s2_gentle_model47900_isaaclab.mp4 | s2_gentle_m47900_sim2sim_mujoco.mp4 |
| s3_rough_l2 | m32800 (JIT) | s3_rough_l2_model32800_isaaclab.mp4 | s3_rough_l2_m32790_sim2sim_mujoco.mp4 |

视频本地路径：`Magicbot_Z1/videos/<run_name>/`

### 9.4 已知问题

- **action_rate L2 崩塌**：s2_gentle 和 s3_rough_l2 尾部出现。使用 L1 替代 L2 可规避
- **超参批量修改**：s2_stable 同时改 6 个超参 → 100 iter 内崩溃。每次只改 1 个
- **地形难度过早引入**：s4_terrain 在机器人未学会走路时引入高难度 → 全失败
- **Checkpoint 不等于最后模型**：model_49999 发散，需在 reward 峰值附近选择

### 9.5 左右关节不对称修复 (2026-05-12)

**问题**：P3/P3b 训练后左右关节角严重不对称（hip_pitch 偏移 0.37 rad, hip_yaw 0.52 rad, knee_pitch 0.39 rad）。

**原因**：reward 函数中 `joint_deviation_l1` 只惩罚各关节偏离默认值，从未约束左右对称。平地（P2）最优步态碰巧对称，但地形训练（P3+）暴露了这个问题。

**方案**：
- 新增 `joint_mirror` reward（`magiclab_rl_lab/tasks/locomotion/mdp/rewards.py` 已有实现），weight=-0.5
- 从 P3b Fine checkpoint 续训 5000 步（`p3b_fine_symmetry` sub-phase）
- 地形配置回退到与 p3 类似（70% flat + 30% random_grid），LR 降至 1e-4

**预期**：左右关节偏移降至 <0.05 rad，步态恢复对称，同时保持地形适应能力。

### 9.6 磁盘清理记录 (2026-05-04)

训练日志：14GB → 5.5GB。s1_flat、s1_flat_retry、s1_stable、s3_rough_l2、s1_highspeed、s3_rough_fail 的中间 checkpoint 已删除（保留 JIT policy）。s2_gentle 和 s3_rough_l1_4gpu 全部保留。

### 9.7 TODO

- [ ] **s4_full_terrain 评估** → 决定下一步
- [ ] **多机器人可视化**：Play/录制视频时同时显示多个机器人（num_envs>1 俯瞰模式）
