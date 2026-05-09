# Standalone Monitor vs Phase Orchestrator 对比

> 两种训练监控/管理方案的架构、能力与适用场景对比。

---

## 1. 定位一句话

| | Standalone Monitor | Phase Orchestrator |
|---|---|---|
| **一句话** | 被动监控现有训练，检测过拟合 | 端到端自动化流水线：配置→训练→监控→回滚→推进→录视频 |
| **类比** | 医生巡房（只看不治） | 全自动 ICU（诊断+给药+调方案） |

---

## 2. 架构对比

### Standalone Monitor (`train_monitor.py`)

```
train_monitor.py (单文件, ~56KB)
├── MonitorConfig          — 阈值/路径配置
├── TensorBoardParser      — 读 TensorBoard event 文件
├── CheckpointAnalyzer     — 遍历 model_*.pt
├── RunState               — 单次运行状态容器
├── OverfittingDetector    — 5 信号过拟合检测
├── BestModelTracker       — 滚动平均最优模型
└── ReportGenerator        — 输出格式化 + 命令生成
```

### Phase Orchestrator (多模块系统)

```
scripts/automation/
├── phase_orchestrator.py  (~49KB) — 主循环：阶段推进、回滚、视频
├── phase_manager.py               — YAML 解析 + 三层合并
├── config_generator.py             — 生成 velocity_env_cfg.py
├── training_launcher.py            — torchrun 启停
├── embedded_monitor.py             — 包装 train_monitor 的检测逻辑
├── state_store.py                  — 原子 JSON 状态持久化
└── ppo_override.py                 — 生成临时 PPO 配置

外部依赖:
├── train_monitor.py  — OverfittingDetector 等核心类
└── train_multigpu.py — 多 GPU 训练入口
```

---

## 3. 功能对比矩阵

| 能力 | Standalone Monitor | Phase Orchestrator |
|------|:-----------------:|:------------------:|
| **过拟合检测** (5 信号) | ✓ | ✓ (复用同一 Detector) |
| **最优模型追踪** | ✓ | ✓ |
| **best_models.json** 输出 | ✓ | ✗ (用 orchestrator_state.json) |
| **训练启动/停止** | ✗ 被动 | ✓ torchrun 启停 |
| **配置生成** (env/agent) | ✗ | ✓ 三层 YAML 合并 |
| **自动回滚** | ✗ 仅报警 | ✓ reward<95%→回退+降 LR |
| **阶段自动推进** | ✗ | ✓ coarse→fine→video→next |
| **视频录制** | ✗ | ✓ 每阶段完成后自动录 |
| **JIT 导出** | ✓ `--auto_export` | ✓ 集成在 pipeline 中 |
| **崩溃恢复** | ✗ 无状态 | ✓ state_store 原子持久化 |
| **单次扫描** (`--once`) | ✓ | ✗ |
| **实时监控** (`--realtime`) | ✓ | ✗ |
| **Dry run** | ✗ | ✓ `--dry-run` |
| **指定阶段启动** (`--start-from`) | ✗ | ✓ |
| **地形自适应阈值** | ✓ flat/gentle/rough | ✓ (YAML 可覆盖) |

---

## 4. 过拟合检测逻辑（共享）

两个系统共用同一套 `OverfittingDetector`，5 个独立信号任一触发即报警：

| # | 信号 | 阈值 | 含义 |
|---|------|------|------|
| 1 | Reward decline | >20% from peak (median) | 策略退化 |
| 2 | Action rate | terrain-specific (flat>-0.8, gentle>-1.0, rough>-1.5) | 关节抖动 |
| 3 | Policy std | < 0.01 | 动作确定性过高 |
| 4 | Value loss | > 100 | 价值函数发散 |
| 5 | Entropy collapse | >95% decline + absolute < 0.5 | 策略坍缩 |

**关键区别**：Monitor 检测后只输出报警；Orchestrator 检测后自动执行停止训练→回滚→重试的完整流程。

---

## 5. 状态持久化

### Standalone Monitor — 无持久状态

```
logs/.../monitor/
├── OVERFITTING_DETECTED    — 标记文件（含详细信号）
└── best_models.json        — 汇总所有 run 的最优模型
```

每次启动重新扫描所有 run，不记忆上一次运行结果。

### Phase Orchestrator — 原子持久化

```
orchestrator_state.json     — 完整 pipeline 状态
├── current_phase_id        — 当前子阶段
├── current_run_dir         — 活跃训练目录
├── best_model_path         — 当前最优 checkpoint
├── best_reward             — 最优 reward
├── rollback_count          — 回滚次数
├── phase_history[]         — 已完成阶段记录
└── started_at              — 启动时间
```

崩溃/手动停止后可通过 `--resume` 从断点继续，无需重做已完成阶段。

---

## 6. 典型使用场景

### 场景 A：手动调参实验 → 用 Standalone Monitor

```bash
# 手动启动训练
ssh rtx "python train.py --task Z1 --run_name my_exp --headless"

# 后台监控，过拟合自动导出 JIT
/gpu-train --monitor --start

# 随时查看状态
/gpu-train --monitor --status
/gpu-train --monitor --realtime    # 实时追踪
```

**适合**：探索性实验、单次训练、需要实时交互、频繁手动干预。

### 场景 B：全流程自动训练 → 用 Phase Orchestrator

```bash
# 一键启动 5 阶段 10 子阶段 pipeline
/gpu-train --automation --start

# 查看整体进度
/gpu-train --automation --status

# 中断后恢复
/gpu-train --automation --resume
```

**适合**：成熟方案的批量训练、过夜/过周末无人值守、需要自动回滚和阶段推进。

### 场景 C：两者并存

```bash
# Orchestrator 运行 pipeline
/gpu-train --automation --start

# 同时启动 Standalone Monitor 做 double-check
/gpu-train --monitor --start

# Monitor 作为独立审计层，不干预 Orchestrator 的训练控制
```

**适合**：对稳定性要求极高的长期训练，双重检测 + 独立报警。

---

## 7. CLI 速查

### Standalone Monitor

```bash
# 单次扫描所有 run
python train_monitor.py --once --terrain gentle

# 实时追踪活跃 run（30s 轮询）
python train_monitor.py --realtime --terrain gentle

# 后台持续监控 + 过拟合自动导出
python train_monitor.py --terrain gentle --poll_interval 120 --auto_export

# 分析单个 run
python train_monitor.py --once --run_dir logs/.../<RUN_DIR>

# Claude 快捷命令
/gpu-train --monitor              # 单次扫描
/gpu-train --monitor --start      # 后台启动
/gpu-train --monitor --status     # 查看状态
/gpu-train --monitor --stop       # 停止
/gpu-train --monitor --realtime   # 实时追踪
/gpu-train --monitor --anal       # 失败分析
```

### Phase Orchestrator

```bash
# 启动完整 pipeline
python phase_orchestrator.py --plan training_plans/z1_5phase_plan.yaml --num-gpus 4 --fresh

# 从指定阶段开始
python phase_orchestrator.py --plan ... --start-from p3_coarse

# 恢复中断的 pipeline
python phase_orchestrator.py --plan ...

# Dry run（验证配置）
python phase_orchestrator.py --plan ... --dry-run

# Claude 快捷命令
/gpu-train --automation --start              # 启动
/gpu-train --automation --status             # 查看进度
/gpu-train --automation --tail               # pipeline 日志
/gpu-train --automation --resume             # 恢复
/gpu-train --automation --stop               # 停止
/gpu-train --automation --dry-run            # 预览
```

---

## 8. 输出文件位置

| 文件 | Standalone Monitor | Orchestrator |
|------|-------------------|--------------|
| 监控日志 | `/tmp/z1_monitor.log` | `/tmp/z1_5phase_pipeline.log` |
| 训练日志 | N/A (外部管理) | `~/magiclab_rl_lab/logs/train_<sub_phase>.log` |
| 状态文件 | `best_models.json` (per log root) | `orchestrator_state.json` (project root) |
| 过拟合标记 | `<run_dir>/monitor/OVERFITTING_DETECTED` | 内嵌在 state JSON |
| JIT 导出 | `<run_dir>/exported/policy.pt` | 同左 |
| 视频 | N/A | `videos/phase_pipeline/<sub_phase>.mp4` |

---

## 9. Orchestrator 的局限

| 局限 | 说明 |
|------|------|
| **必须预写 YAML** | 10 个子阶段的 obs/reward/terrain/LR 全部预设，没写就没法启动 |
| **跑起来改不了** | 只能停→改 YAML→重来，不能中途调参 |
| **无实时交互** | 没有 `--realtime` 模式，只能看日志 |
| **探索阶段浪费** | reward 权重、curriculum 阶段数都不确定时，写 YAML 纯属浪费时间 |
| **调试困难** | 多模块调用链长，出错要追 phase_manager→config_generator→training_launcher→embedded_monitor |

## 10. 真实工作流

### 日常迭代循环（Standalone Monitor 主力）

```
改代码 → /gpu-train --start --resume → /gpu-train --monitor --realtime
   ↑                                          ↓
   └──── 不行就 kill，改完重来 ←── reward 不涨 / 过拟合
```

这个快速迭代循环是 Orchestrator 做不到的。

### 成熟方案批量训练（Orchestrator 上场）

```
YAML 计划调通 → /gpu-train --automation --start → 过夜/过周末无人值守
                                                      ↓
                                              自动完成 10 子阶段
                                              自动回滚 + 录视频
                                              崩溃后 --resume 继续
```

**前提**：reward 权重、curriculum 阶段数、terrain 类型都已经用 Standalone Monitor 手动验证过了。

## 11. 决策一句话

- **还在调参** → Standalone Monitor（`--monitor --realtime`）
- **方案已定，跑量** → Orchestrator（`--automation --start`）

> 顺带说明：代码叫 `phase_orchestrator.py`，Claude 命令用 `--automation`，两者是同一个东西。
> 命名差异是因为 Claude skill 面向用户操作语义（"自动化"），代码面向技术语义（"编排器"）。

---

*最后更新: 2026-05-08*
