# MagicBot Z1 12DOF — RL Locomotion 2-min Share

**总时长**: 约 2 分钟
**页数**: 5 页

---

## Page 1: 标题 — MagicBot Z1 双足机器人 RL 运动控制 (15s)

- 标题：MagicBot Z1 12DOF 双足机器人强化学习运动控制
- 一句话：基于 Isaac Lab + PPO 训练 12 自由度双足机器人在多样化地形上行走
- 配图：Z1 机器人 MuJoCo 渲染图
  - ![Z1 MuJoCo](sources/z1_mujoco.png)

---

## Page 2: 系统架构 (20s)

- Two-Platform 架构
  - RTX 6000 远程训练服务器：8×GPU, 16384 并行环境, Isaac Lab + rsl_rl
  - 本地 Windows：MuJoCo Sim2Sim 验证 + 分析
- 5-Phase 自动化训练 Pipeline
  - P1 平地 → P2 平地精调 → P3 缓坡 → P3b 中等地形 → P4 复杂地形
- 关键技术：Curriculum Learning + 自动 Phase 切换 + 过拟合检测

---

## Page 3: 训练策略 — Reward & PPO (25s)

### Reward 设计 (10+ 项)

**激励项**:
- XY 速度跟踪 (w=1.0)
- 角速度跟踪 (w=0.5)
- 存活奖励 (w=0.15)
- 足部接触时序 (w=0.5)
- 足部摆动高度 (w=1.0)

**惩罚项**:
- Z 轴速度 (w=-2.0)
- 身体姿态 (w=-5.0)
- 基座高度 (w=-10.0)
- 能量消耗 (w=-2e-5)
- 足部滑动 (w=-0.2)
- 动作变化率 (w=-0.1)

### PPO 配置
- 网络: MLP (32, 32), separate actor-critic
- Learning rate: 3e-4, Entropy: 0.01, GAE λ=0.95

- 配图：Reward 分解图
  - ![Reward Decomposition](sources/reward_decomposition.png)

---

## Page 4: 训练成果 (30s)

### Curriculum 学习曲线
- 多 Phase Reward 趋势：P1→P2→P3b 逐步提升
  - ![Curriculum Trends](sources/curriculum_reward_trends.png)

### 关键指标
| Phase | 最佳 Reward | 步态距离 | 摔倒率 | Sim2Sim |
|-------|-----------|---------|--------|---------|
| P2 Fine | 49.68 | 4.0m/10s | 0% | OK |
| 本地测试 | — | 12m/25s | 0.06% | OK |

### Demo GIF
- P1→P2 训练演示动画
  - ![Pipeline Demo](sources/pipeline_demo_frame.png)

---

## Page 5: Demo & 下一步 (30s)

### Sim2Sim 验证
- 平地策略 (P1/P2) → MuJoCo 成功迁移，零摔倒
- 地形策略 (P3/P3b) → Sim2Sim Gap，MuJoCo 中摔倒或冻结
  - ![Sim2Sim Broken](sources/sim2sim_broken_frame.png)

### 现场演示
- 本地 MuJoCo 键盘实时操控

### 下一步
1. 解决 Sim2Sim Gap（观测空间/物理差异）
2. Sim2Real 真机部署
3. 复杂地形自适应行走
