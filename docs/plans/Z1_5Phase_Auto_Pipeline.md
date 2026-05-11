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

---

## 训练历史与结果

> 最后更新: 2026-05-11 16:15 (GMT+8)
> 当前状态: **p3_fine RUNNING** (pipeline v2, velocity-balanced reward)

### 总览

```
Timeline (GMT+8):
05-06  pipeline v1 启动: p1 → p2 → p3 (失败)
05-07  pipeline v1 继续: p3 反复重试 → p4 → p5 (old_v1)
05-08  pipeline v1 完成: p3_v2 → p3_fine → p4 → p5 (v2)
05-09  补充 p3b phase: p3b_coarse → p3b_fine
05-10  pipeline v2 重启: p3_coarse reward 重调 → velocity-balanced
05-11  p3_coarse 完成 → p3_fine RUNNING (当前)
```

### P1 — Flat Bootstrap

| # | 时间 | Run Dir | 状态 | Best Model | Reward | 备注 |
|---|------|---------|------|-----------|--------|------|
| 1 | 05-06 15:47 | `p1_coarse` | OVERFITTING | model_2900 | 15.61 | 熵坍缩 97.5%，首次 pipeline |
| 2 | 05-06 17:40 | `p1_fine` | OVERFITTING | model_2800 | 5.54 | reward 反降 48.7% |
| 3 | 05-07 11:40 | `p1_coarse` | FAILED | model_0 | — | 短命 run |
| 4 | 05-07 17:01 | `p1_coarse` | FAILED | model_0 | — | 短命 run |

**Best**: p1_coarse model_2900 (reward 15.61)
**本地模型**: checkpoint + jit_policy ✓

### P2 — Flat Velocity Tracking

| # | 时间 | Run Dir | 状态 | Best Model | Reward | 备注 |
|---|------|---------|------|-----------|--------|------|
| 1 | 05-06 18:49 | `p2_coarse` | OVERFITTING | model_2917 | 41.46 | zombie data |
| 2 | 05-06 19:33 | `p2_fine` | OVERFITTING | model_3168 | 48.30 | zombie data, **全局最高 reward** |
| 3 | 05-07 03:00 | `p2_coarse` | FAILED | model_100 | — | 重试 |
| 4 | 05-07 03:10 | `p2_fine` | COMPLETED | model_3000 | — | pipeline v1 |
| 5 | 05-08 06:38 | `p2_fine` | FAILED | model_100 | — | 短命 run |

**Best**: p2_fine model_3168 (reward 48.30) — 全局最佳 flat 地形成绩
**本地模型**: checkpoint + jit_policy ✓

### P3 — Gentle Terrain（最多波折的 phase）

#### v1 — 原版 standing-biased (alive=0.5, stand_still=0)

| # | 时间 | Run Dir | 状态 | Best Model | Reward | 备注 |
|---|------|---------|------|-----------|--------|------|
| 1 | 05-07 03:56 | `p3_coarse_old_v1` | OVERFITTING | model_7700 | — | bad_ori 11.9%, peak 37.19 |
| 2 | 05-07 11:23 | `p3_coarse_old_v1` | CONTINUED | model_8000 | — | 继续训练 |
| 3 | 05-07 11:53 | `p3_coarse_old_v1` | OVERFITTING | model_11700 | — | 最终 93 个 checkpoint |
| 4 | 05-07 15:15 | `p3_fine_old_v1` | OVERFITTING | model_10300 | — | reward 降 23.7% |
| 5 | 05-08 06:50 | `p3_coarse_old_v1` | FAILED | model_3200 | — | 短命 run |
| 6 | 05-08 07:00 | `p3_coarse_old_v1` | FAILED | model_0 | — | 短命 run |
| 7 | 05-08 07:03 | `p3_coarse_old_v1` | FAILED | model_0 | — | 短命 run |

**问题**: flat→terrain 跨度大，penalties 太紧，entropy 不够；且 standing-biased 导致 MuJoCo 走不动

#### v2 — 速度平衡版 (alive=0.25, stand_still=-2.0)

| # | 时间 | Run Dir | 状态 | Best Model | Reward | 备注 |
|---|------|---------|------|-----------|--------|------|
| 1 | 05-08 07:09 | `p3_coarse_v2` | OVERFITTING | model_11600 | — | 87 个 checkpoint |
| 2 | 05-08 12:45 | `p3_fine` | OVERFITTING | model_10555 | 36.02 | peak 41.17, reward 降 27.6% |

**问题**: 仍然过拟合，但 MuJoCo 可行走

#### Pipeline v2 — 从头重跑 (resume from p2_fine best)

| # | 时间 | Run Dir | 状态 | Best Model | Reward | 备注 |
|---|------|---------|------|-----------|--------|------|
| 1 | 05-10 19:13 | `p3_coarse` | FAILED | model_0 | — | 短命 run |
| 2 | 05-10 19:16 | `p3_coarse` | KILLED | model_5800 | 36.37 | standing-biased, 手动停掉 |
| 3 | 05-11 04:42 | `p3_coarse` | OVERFITTING | model_6000 | — | velocity-balanced 重跑, 10400 iter |
| **4** | **05-11 08:33** | **`p3_fine`** | **RUNNING** | model_6300 | — | **当前运行中** |

**Best so far**: p3_coarse model_5800 (reward 36.37)
**本地模型**: p3_coarse checkpoint + jit_policy ✓

### P3b — Intermediate Terrain（补充 phase）

> 在 p3 与 p4 之间插入的中间阶段，使用 gentle terrain。

| # | 时间 | Run Dir | 状态 | Best Model | Reward | 备注 |
|---|------|---------|------|-----------|--------|------|
| 1 | 05-09 03:22 | `p3b_coarse` | OVERFITTING | model_10600 | 29.08 | bad_ori 16.6% |
| 2 | 05-09 06:09 | `p3b_fine` | OVERFITTING | model_10800 | — | — |

**Best**: p3b_coarse model_10600 (reward 29.08)
**本地模型**: 两个 sub-phase 均有 checkpoint + jit_policy ✓

### P4 — Rough Terrain

#### v1 — old pipeline (05-08)

| # | 时间 | Run Dir | 状态 | Best Model | 备注 |
|---|------|---------|------|-----------|------|
| 1 | 05-08 13:03 | `p4_coarse` | COMPLETED | model_10700 | old pipeline |
| 2 | 05-08 13:16 | `p4_fine` | COMPLETED | model_10800 | old pipeline |

#### v2 — pipeline v2 (05-09)

| # | 时间 | Run Dir | 状态 | Best Model | 备注 |
|---|------|---------|------|-----------|------|
| 1 | 05-09 09:50 | `p4_coarse` | COMPLETED | model_15900 | 52 个 checkpoint |
| 2 | 05-09 14:00 | `p4_fine` | COMPLETED | model_15000 | 23 个 checkpoint |

**Best**: p4_coarse model_15900

### P5 — Full Terrain + Polish

#### v1 — old pipeline (05-07)

| # | 时间 | Run Dir | 状态 | Best Model | 备注 |
|---|------|---------|------|-----------|------|
| 1 | 05-07 18:11 | `p5_coarse_old_v1` | COMPLETED | model_10300 | — |
| 2 | 05-07 18:33 | `p5_fine_old_v1` | COMPLETED | model_10400 | — |

#### v2 — pipeline v2 (05-08)

| # | 时间 | Run Dir | 状态 | Best Model | 备注 |
|---|------|---------|------|-----------|------|
| 1 | 05-08 13:28 | `p5_coarse` | COMPLETED | model_11000 | — |
| 2 | 05-08 13:46 | `p5_fine` | COMPLETED | model_11100 | — |

**Best**: p5_coarse model_11000

### 非 Pipeline 历史 Run（手动训练）

| 时间 | Run Dir | Best Model | 备注 |
|------|---------|-----------|------|
| 04-30 04:53 | `s1_flat` | — | 早期手动 run，无 checkpoint |
| 05-01 04:50 | `s2_gentle` | model_47900 | 501 个 checkpoint，早期最完整 run |
| 05-01 07:04 | `s3_rough_l2` | — | L2 action rate，无 checkpoint |
| 05-04 16:56 | `s4_full_terrain` | model_15000 | 101 个 checkpoint |

### 当前 Pipeline 状态 (2026-05-11 16:15)

```
Pipeline Orchestrator: PID 768226 ● Running
Training Process:      PID 803323 (4× RTX 6000)
当前 Sub-Phase:         p3_fine → 2026-05-11_08-33-26_p3_fine
Latest Checkpoint:     model_6300.pt

Stage History:
  ✅ p2_fine_resume   → COMPLETE (reward 36.37)
  ✅ p3_coarse         → OVERFITTING → advanced to p3_fine (best: model_6000)
  🔄 p3_fine           → RUNNING (current)

Remaining:
  p3_fine → p4_coarse → p4_fine → p5_coarse → p5_fine
```

### 全局 Best Model 汇总

| Phase | Sub-Phase | Best Model | Reward | 地形 | 状态 |
|-------|-----------|-----------|--------|------|------|
| p1 | p1_coarse | model_2900 | 15.61 | flat | OVERFITTING |
| p2 | p2_fine | model_3168 | **48.30** | flat | OVERFITTING |
| p3 | p3_coarse (v2) | model_5800 | 36.37 | gentle | KILLED → resumed |
| p3b | p3b_coarse | model_10600 | 29.08 | gentle | OVERFITTING |
| p4 | p4_coarse (v2) | model_15900 | — | rough | COMPLETED |
| p5 | p5_coarse (v2) | model_11000 | — | rough | COMPLETED |

**Note**: p4/p5 的 reward 数据缺失（历史 pipeline 未记录），可通过 `train_monitor.py` 补查。

### Bug 修复记录

| 日期 | Bug | Root Cause | Fix |
|------|-----|-----------|-----|
| 05-11 | Orchestrator monitor TensorBoard stale | `_find_latest_run_dir` resume 时立即匹配到旧目录 | 添加 `launch_time` + `min_ctime` 过滤 |
| 05-11 | Resume 创建新目录但 state 指向旧目录 | `train_multigpu.py` 用 `datetime.now()` 生成新 log_dir | 更新 state + 代码修复 `min_ctime` |
