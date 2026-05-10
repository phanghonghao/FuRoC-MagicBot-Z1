---
title: MagicBot Z1 12DOF
subtitle: 基于强化学习的双足机器人运动控制
author: MagicLab
date: 2026.05
---

# 标题页

## MagicBot Z1 12DOF

基于强化学习的双足机器人运动控制

Isaac Lab + PPO | 16384 envs | Curriculum Learning

![](sources/z1_mujoco.png)

---

# 系统架构

## RTX 6000 远程训练

- 8x RTX 6000 GPU (85GB VRAM)
- 16384 并行仿真环境
- Isaac Lab + rsl_rl PPO
- 自动化 5-Phase Pipeline

## 本地分析 & 验证

- MuJoCo Sim2Sim 验证
- 学习曲线 & 过拟合分析
- 键盘实时操控测试

---

# 5-Phase Curriculum

## 训练阶段

1. **P1** 平地粗训
2. **P2** 平地精调
3. **P3** 缓坡地形
4. **P3b** 中等地形
5. **P4** 复杂地形

---

# 训练策略 — Reward 设计 & PPO

## 激励项

| Reward 项 | 权重 |
|-----------|------|
| XY 速度跟踪 | w=1.0 |
| 角速度跟踪 | w=0.5 |
| 存活奖励 | w=0.15 |
| 足部接触时序 | w=0.5 |
| 足部摆动高度 | w=1.0 |

## 惩罚项

| Penalty 项 | 权重 |
|------------|------|
| Z 轴速度 | w=-2.0 |
| 身体姿态偏移 | w=-5.0 |
| 基座高度偏差 | w=-10.0 |
| 能量消耗 | w=-2e-5 |
| 足部滑动 | w=-0.2 |
| 动作变化率 | w=-0.1 |

![](sources/reward_decomposition.png)

## PPO 配置

| 参数 | 值 |
|------|-----|
| 网络 | MLP (32, 32), Actor-Critic |
| Learning Rate | 3e-4 |
| Entropy Coeff | 0.01 |
| GAE λ | 0.95 |

---

# 训练成果

## Curriculum 学习曲线

![](sources/curriculum_reward_trends.png)

## P1→P2 训练演示

![](sources/pipeline_demo.gif)

---

# 关键指标

| 指标 | 数值 |
|------|------|
| P2 最佳 Reward | 49.68 |
| 步态距离 (Sim2Sim) | 4.0m / 10s |
| Sim2Sim 摔倒率 | 0% |
| 本地测试距离 | 12.0m / 25.5s |
| 本地摔倒率 | 0.06% |

---

# Sim2Sim 验证 & 下一步

## Sim2Sim 验证结果

| Phase | 地形 | MuJoCo 迁移 |
|-------|------|-------------|
| P1 Fine | 平地 | OK |
| P2 Fine | 平地 | OK |
| P3 Fine | 缓坡 | 20 falls |
| P3b Fine | 中等 | 冻结 |

## P3 Sim2Sim 失败

![](sources/sim2sim_broken.gif)

---

# 核心发现 & 下一步

## 核心发现

- 平地策略 → MuJoCo 迁移成功
- 地形策略 → Sim2Sim Gap
- 原因：物理引擎差异（接触力/摩擦）

## 下一步计划

1. 解决 Sim2Sim Gap（观测空间对齐 + 物理参数标定）
2. Sim2Real 真机部署（Domain Randomization + 降阶）
3. 复杂地形自适应行走（越障、楼梯、不平地面）
