# Z1 12DOF 5-Phase Automated Training Pipeline

> 5 阶段全自动 phase-based 训练 pipeline，每个 phase 含 coarse + fine 两个 sub-phase，
> 自动检测过拟合、自动回滚、自动录视频。全程使用 action_rate_l1。

## Pipeline 总览

```
p1 (flat, bootstrap)          从头训练
  p1_coarse → p1_fine → 录视频
    ↓ resume from best checkpoint
p2 (flat, velocity tracking)
  p2_coarse → p2_fine → 录视频
    ↓
p3 (gentle terrain)
  p3_coarse → p3_fine → 录视频
    ↓
p4 (rough terrain)
  p4_coarse → p4_fine → 录视频
    ↓
p5 (full terrain + polish)
  p5_coarse → p5_fine → 录视频 → 最终部署模型
```

共 **10 个 sub-phase**，总训练量 ~210K iterations。

## 训练配置

- **GPU**: RTX 6000 × 4 卡 (torchrun)
- **起点**: 从头训练
- **Env**: 4096 environments
- **Action rate**: 始终用 `action_rate_l1`（永不用 L2）

## 各 Phase 参数详解

### 粗调 vs 细调 通用区别

| 参数 | 粗调 (coarse) | 细调 (fine) |
|------|---------------|-------------|
| learning_rate | 1e-3 | 5e-4 (p1-p3) / 3e-4 (p4) / 1e-4 (p5) |
| entropy_coef | 0.01-0.012 | 0.005-0.008 |
| 惩罚权重 | 松 (小惩罚) | 紧 (大惩罚) |
| alive | 0.3-0.5 | 0.1-0.15 |
| action_rate | -0.02~-0.04 | -0.04~-0.08 |

### Phase 1: Flat — Bootstrap (10K iter)

**目标**: 让机器人学会站立和基本平衡，不做速度追踪。

| 参数 | p1_coarse | p1_fine |
|------|-----------|---------|
| 地形 | plane | plane |
| 速度范围 | vx/vy/vyaw [-0.1, 0.1] | 同左 |
| LR | 1e-3 | 5e-4 |
| entropy | 0.01 | 0.008 |
| alive | 0.5 | 0.3 |
| base_height | -5.0 | -8.0 |
| flat_orientation | -2.0 | -4.0 |
| action_rate | -0.02 | -0.04 |
| stand_still | 0 | -2.0 |
| track_lin_vel | 0 (关闭) | 0 (关闭) |

### Phase 2: Flat — Velocity Tracking (20K iter)

**目标**: 引入速度追踪，从低速逐步扩展到全速范围。

| 参数 | p2_coarse | p2_fine |
|------|-----------|---------|
| 地形 | plane | plane |
| 速度范围 | vx[-0.3,0.5] vy[-0.3,0.3] | vx[-0.5,1.0] vy[-0.5,0.5] |
| LR | 1e-3 | 5e-4 |
| entropy | 0.01 | 0.008 |
| track_lin_vel | 1.5 | 2.0 |
| track_ang_vel | 0.75 | 1.0 |
| base_height | -8.0 | -10.0 |
| flat_orientation | -4.0 | -5.0 |
| action_rate | -0.03 | -0.05 |
| stand_still | 0 | -3.5 |

### Phase 3: Gentle Terrain (30K iter)

**目标**: 引入地形，70% flat + 30% random_grid。

#### Stage 3 v1 (原版 — 过拟合 + 跌倒多)

> p3_coarse: bad_ori 11.9%, reward peak 37.19 → 30.33 (↓18.7%)
> p3_fine: bad_ori 13.6%, reward peak 34.02 → 25.95 (↓23.7%)
> 根因: flat→terrain 跨度太大，penalties 太紧，entropy 不够

| 参数 | p3_coarse | p3_fine |
|------|-----------|---------|
| 地形 | 70% flat + 30% grid [0,0.4] | 70% flat + 30% grid [0,0.5] |
| LR | 5e-4 | 3e-4 |
| entropy | 0.01 | 0.008 |
| track_lin_vel | 1.5 | 2.0 |
| alive | 0.3 | 0.15 |
| flat_orientation | -5.0 | -7.0 |
| action_rate | -0.04 | -0.05 |
| feet_clearance | 1.0 | 1.0 |
| base_height | -8.0 | -10.0 |
| undesired_contacts | 0 | -1.5 |

#### Stage 3 v2 (当前 — 速度平衡：降低 alive，加 stand_still 惩罚)

> 站立优先版 alive=0.5 + stand_still=0 导致 policy 学会"站着不动"，
> MuJoCo 里完全走不动。改为速度平衡版：降低 alive，恢复 stand_still 惩罚。
> Resume from model_5800.pt (peak reward 36.37 @iter 5830)。

| 参数 | p3_coarse | p3_fine | 变更说明 |
|------|-----------|---------|----------|
| 地形 | 70% flat + 30% grid **[0,0.25]** | 70% flat + 30% grid **[0,0.35]** | ↓ 地形难度 |
| LR | **3e-4** | **2e-4** | ↓ 更慢收敛 |
| entropy | **0.015** | **0.012** | ↑ 更多探索 |
| track_lin_vel | **2.0** | 2.0 | ↑ 加强速度追踪 |
| alive | **0.25** | **0.15** | ↓ 降低站立奖励 |
| flat_orientation | **-4.0** | -5.0 | 稍收紧 |
| base_height | **-6.0** | **-8.0** | 稍收紧 |
| action_rate | -0.04 | -0.05 | — |
| feet_clearance | **0.8** | **1.0** | ↑ 鼓励抬脚 |
| stand_still | **-2.0** | **-3.5** | ↑ 惩罚不动 |
| undesired_contacts | 0 | -1.5 | — |

### Phase 4: Rough Terrain (50K iter)

**目标**: 完整地形类型，30% flat + 30% grid + 20% stairs + 10% gap + 10% boxes。

| 参数 | p4_coarse | p4_fine |
|------|-----------|---------|
| 地形 | 全类型 difficulty [0,0.8] | 全类型 difficulty [0,1.0] |
| LR | 5e-4 | 3e-4 |
| entropy | 0.012 (高探索) | 0.008 |
| track_lin_vel | 1.5 | 2.0 |
| alive | 0.3 | 0.15 |
| flat_orientation | -5.0 | -7.0 |
| action_rate | -0.04 | -0.06 |
| feet_slide | -0.3 | -0.4 |
| undesired_contacts | -1.5 | -2.0 |
| energy | -3e-5 | -5e-5 |

### Phase 5: Full Terrain + Polish (60K iter)

**目标**: 全类型全难度，强域随机化，最终打磨。

| 参数 | p5_coarse | p5_fine |
|------|-----------|---------|
| 地形 | 全类型全难度 [0,1.0] | 同左 |
| LR | 3e-4 | 1e-4 |
| entropy | 0.01 | 0.005 |
| track_lin_vel | 1.5 | 2.0 |
| alive | 0.15 | 0.1 |
| flat_orientation | -7.0 | -7.0 |
| action_rate | -0.05 | -0.08 |
| feet_slide | -0.4 | -0.5 |
| energy | -5e-5 | -8e-5 |
| joint_acc | -1e-6 | -2e-6 |

## 自动回滚机制

每个 sub-phase 结束时：

```
if best_reward < starting_reward × 0.95:
    → 记录 "rollback"，丢弃本轮结果
    → 用 starting checkpoint 重试 (LR × 0.5)
    → 最多重试 1 次
else:
    → 正常推进，用本轮 best checkpoint 作为下一个 sub-phase 的起点
```

## 文件结构

### 新建文件

```
scripts/automation/
  ├── phase_orchestrator.py    # 新 orchestrator，两层循环 (phase → sub_phase)
  ├── phase_manager.py         # 解析 YAML，三层合并 (defaults → phase → sub_phase)
  ├── config_generator.py      # 从参数 dict 生成 velocity_env_cfg.py
  └── ppo_override.py          # 生成临时 PPO config (覆盖 LR/entropy 等)

training_plans/
  └── z1_5phase_plan.yaml      # 5-phase 完整参数
```

### 修改文件

```
scripts/rsl_rl/cli_args.py     # + --agent_cfg 参数
scripts/rsl_rl/train.py        # + _load_agent_cfg() 动态加载 PPO config
scripts/automation/state_store.py       # + phase_id / starting_reward / phase_history
scripts/automation/training_launcher.py # + agent_cfg 传递给子进程
```

### 不动的文件

```
embedded_monitor.py            # 过拟合检测逻辑直接复用
train_monitor.py               # 参数化设计，不需要改
rsl_rl_ppo_cfg.py              # 基础配置保留，override 通过生成文件
```

## 使用方法

```bash
# 完整 pipeline (4 卡)
python -m automation.phase_orchestrator \
    --plan training_plans/z1_5phase_plan.yaml \
    --num-gpus 4

# Dry run (打印所有 10 个 sub-phase 的参数，不执行)
python -m automation.phase_orchestrator \
    --plan training_plans/z1_5phase_plan.yaml --dry-run

# 从特定 sub-phase 开始
python -m automation.phase_orchestrator \
    --plan training_plans/z1_5phase_plan.yaml \
    --start-from p3_coarse --num-gpus 4

# 忽略已保存状态，从头开始
python -m automation.phase_orchestrator \
    --plan training_plans/z1_5phase_plan.yaml --fresh --num-gpus 4
```

## 每个 Sub-Phase 执行流程

```
1. config_generator 生成 velocity_env_cfg.py
2. 生成 PPO override config
3. 备份当前 env config → 替换为生成的
4. 解析 checkpoint (resume from 上一个 best)
5. 4 卡 torchrun 启动训练
6. embedded_monitor 监控过拟合
7. 检测到过拟合 → 停止 → 记录 best checkpoint
8. 回滚判断：best_reward < starting_reward × 0.95 → 重试 (LR×0.5)
9. 录制仿真视频 → videos/phase_pipeline/{sub_phase_id}.mp4
10. 推进到下一个 sub-phase
```

## 监控阈值

| Phase | action_rate_threshold | min_iterations |
|-------|----------------------|----------------|
| p1    | -0.6                 | 2000           |
| p2    | -0.7 ~ -0.8          | 3000           |
| p3    | -1.0                 | 4000           |
| p4    | -1.5                 | 5000           |
| p5    | -1.5                 | 8000           |

## 设计原则

1. **永远用 action_rate_l1** — L2 会导致 s3/s4 过拟合崩塌
2. **每个 phase 有独立奖励权重** — 不再所有阶段用相同配置
3. **粗调先宽后严** — 粗调松惩罚让模型自由探索，细调紧惩罚收敛
4. **地形渐进** — plane → gentle → rough → full，避免 domain gap 过大
5. **自动回滚** — 避免「越训越差」的问题，自动丢弃失败的 sub-phase
6. **生成而非手写配置** — 10 个 sub-phase 通过 config_generator 动态生成，避免手动维护 10 份文件
