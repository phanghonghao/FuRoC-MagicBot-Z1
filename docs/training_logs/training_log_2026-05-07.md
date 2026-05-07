# Z1 12DOF Training Log

> RTX 6000D Server | phh@192.168.120.155 | Server timezone: UTC (+0)
> All timestamps below: **CST (GMT+8)**

---

## Run 1: 5-Phase Pipeline (2026-05-06 ~ 2026-05-07)

**Plan:** `training_plans/z1_5phase_plan.yaml`
**GPU:** 4x RTX 6000D (torchrun, port 29502)
**num_envs:** 4096/GPU

### Timeline (CST)

| Time | Event |
|------|-------|
| 05/06 23:47 | Pipeline 启动, p1_coarse 开始 |
| 05/07 00:56 | p1_coarse 完成 (best: model_2900) |
| 05/07 01:40 | p1_fine 开始 |
| 05/07 01:44 | p1_fine 完成 (best: model_2800) |
| 05/07 01:51 | p2_coarse attempt 1 — torchrun 启动 |
| 05/07 01:55 | p2_coarse attempt 1 — 4min 超时, 放弃 (**未 kill 进程**) |
| 05/07 01:59 | p2_coarse attempt 2 — 又一个 torchrun 抢同一端口+GPU |
| 05/07 02:03 | p2_coarse attempt 2 — 超时 |
| 05/07 02:07 | p2_coarse attempt 3 — 第三个 torchrun |
| 05/07 02:11 | p2_coarse attempt 3 — 超时 |
| 05/07 02:13 | Pipeline 停止 (3 个 zombie torchrun 仍在运行) |
| 05/07 02:49 | p2_coarse 训练实际开始 (zombie 抢到 GPU) |
| 05/07 03:27 | p2_coarse 最后 checkpoint model_3900 |
| 05/07 03:33 | p2_fine 启动 |
| 05/07 ~04:00 | p2_fine 被 kill (signal 15) |
| 05/07 ~04:14 | p3_coarse 启动, 也被 kill |

### Sub-phase Details

#### P1: Flat — Bootstrap

**p1_coarse** (`2026-05-06_15-47-12_p1_coarse`)
- Iterations: 0 → 6 / 5000 (pipeline 提前终止)
- Reward: -0.43 → -0.41
- Best: model_2900.pt
- Status: COMPLETED (pipeline 判定收敛)

**p1_fine** (`2026-05-06_17-40-13_p1_fine`)
- Iterations: 2700 → 2885 / 7700
- Reward: ? → 2.90
- Speed: ~209k steps/s, 1.85s/iter
- Metrics:

| Metric | Value |
|--------|-------|
| mean_reward | 2.90 |
| episode_length | 950.6 |
| time_out | 92.9% |
| bad_orientation | 7.1% |
| error_vel_xy | 0.414 |
| error_vel_yaw | 0.879 |
| action_noise_std | 0.40 |
| lin_vel_cmd | 0.10 |
| ang_vel_cmd | 0.10 |

- Best: model_2800.pt
- Status: COMPLETED

#### P2: Flat — Velocity Tracking

**p2_coarse** (`2026-05-06_18-49-40_p2_coarse`) — *Pipeline 失控后 zombie 进程自行运行*
- Iterations: 2800 → 3929 / 12800
- Reward: -0.81 → 33.02
- Speed: ~198k steps/s, 2.01s/iter
- Elapsed: 37m50s
- Checkpoints: 12 files (model_2800 ~ model_3900)

| Metric | Start | End |
|--------|-------|-----|
| mean_reward | -0.81 | 33.02 |
| episode_length | ? | 980.4 |
| time_out | ? | 97.4% |
| bad_orientation | ? | 2.6% |
| error_vel_xy | ? | 0.340 |
| error_vel_yaw | ? | 0.749 |
| action_noise_std | ? | 0.60 |
| lin_vel_cmd | ? | 1.00 |
| ang_vel_cmd | ? | 0.30 |

- Best: model_3900.pt (last saved)
- Status: KILLED (signal 15)

**p2_fine** (`2026-05-06_19-33-51_p2_fine`) — *非 pipeline 控制*
- Iterations: 2900 → 3667 / 12900
- Reward: -0.50 → 38.32
- Speed: ~? steps/s, 2.01s/iter
- Elapsed: 25m54s
- Checkpoints: 8 files

| Metric | Value |
|--------|-------|
| mean_reward | 38.32 |
| time_out | 97.0% |
| bad_orientation | 3.0% |
| error_vel_xy | ? |
| error_vel_yaw | 0.784 |
| action_noise_std | ? |

- Best: model_3600.pt (last saved)
- Status: KILLED (signal 15)

### Pipeline Failure Root Cause

1. **Orchestrator retry 时未 kill 前一个 torchrun** — 3 个进程争抢 GPU 和 port 29502
2. **4 分钟 run directory 等待超时太短** — 多 GPU Isaac Sim 初始化慢, 加上 zombie 进程抢资源
3. **所有 retry 写入同一 log 文件** — `train_p2_coarse.log` 被覆盖

### Current State

- Pipeline: STOPPED
- 训练进程: 无 (所有进程已死亡)
- State file: `orchestrator_state.json` (停在 p2_coarse, pending)
- 最佳 checkpoint: p2_fine/model_3600.pt (reward: 38.32)

---

## Bugfix: Retry Zombie Kill Logic (2026-05-07)

**文件**: `scripts/automation/phase_orchestrator.py`

### Fix A: `_start_sub_phase()` — run_dir 找不到时 kill 进程
- **问题**: run directory 等待超时后直接 return，torchrun 变 zombie
- **修复**: return 前 `graceful_stop(self._proc.pid)` + 清除 `self._proc`

### Fix B: `_handle_failure()` — retry 前 kill 旧进程
- **问题**: retry 时直接 reset state，未 kill 前一个 torchrun，导致多个进程争抢 GPU
- **修复**: `retry_count += 1` 前检查 `training_pid` 是否存活，存活则 `graceful_stop`

### Fix C: `config_generator.py` — MeshRandomGridTerrainCfg 参数映射
- **问题1**: `RandomGridTerrainCfg` 在新版 IsaacLab 已重命名为 `MeshRandomGridTerrainCfg`
- **问题2**: 新 API 不接受 `difficulty_range`，需要 `grid_width` + `grid_height_range`
- **问题3**: `grid_width=0.5` 整除 terrain size 8.0 导致 border_width=0
- **修复**: config_generator 对 `RandomGridTerrainCfg` 做参数转换: `difficulty_range` → `grid_height_range`, `grid_width=0.6`

---

## Pipeline Resume: p2_coarse (2026-05-07)

| Time | Event |
|------|-------|
| 05/07 ~10:30 | Orchestrator bugfix 部署 |
| 05/07 ~10:35 | `--start-from p2_coarse --fresh` 启动 pipeline (PID=2901187) |
| 05/07 ~10:37 | p2_coarse: 找到旧 zombie 数据, overfitting 判定, 跳过 |
| 05/07 ~10:44 | p2_fine: 同上, 旧 zombie 数据, 跳过 |
| 05/07 ~10:49 | p2 COMPLETE, 推进到 p3_coarse |
| 05/07 ~10:53 | p3_coarse 失败: `RandomGridTerrainCfg` 不存在 → 修复 config_generator |
| 05/07 ~11:02 | p3_coarse 失败: `difficulty_range` 参数不匹配 → 修复参数映射 |
| 05/07 ~11:10 | p3_coarse 失败: `grid_width=0.5` 整除 terrain size → 改为 0.6 |
| 05/07 ~11:56 | **p3_coarse 成功启动** (PID=2966244, run_dir=`2026-05-07_03-56-16_p3_coarse`) |
| 05/07 ~11:59 | 验证: iter=65, reward=4.23, HEALTHY, GPU 0-3 ~50% 利用率 |

**命令**:
```bash
rm -f orchestrator_state.json && \
nohup python -u scripts/automation/phase_orchestrator.py \
  --plan training_plans/z1_5phase_plan.yaml \
  --start-from p3_coarse --fresh --num-gpus 4 --poll-interval 120 \
  > /tmp/z1_5phase_pipeline.log 2>&1 & echo PID=$!
```

**状态**: p3_coarse 训练中 (reward 4.23, HEALTHY)

---

## Isaac Sim 视频录制失败分析 (2026-05-07)

### 目标
为 p1/p2 阶段的 best model 补录 Isaac Lab 视频 (共 4 个):

| 阶段 | Checkpoint | Reward |
|------|-----------|--------|
| p1_coarse | model_2700.pt | ~15.61 (peak@2690) |
| p1_fine | model_2800.pt | 2.90 |
| p2_coarse | model_2900.pt | ~42.68 (peak, zombie data) |
| p2_fine | model_3200.pt | ~48.30 (best@3168, zombie data) |

### 结果

| 视频 | 状态 | 说明 |
|------|------|------|
| p1_coarse | **成功** (200帧, 20MB) | 旧进程在训练前碰巧完成 |
| p1_fine | 失败 | 进程静默死亡 |
| p2_coarse | 失败 | 进程静默死亡 |
| p2_fine | 失败 | 进程静默死亡 |

### 失败原因

**Isaac Sim 渲染管线会占用全部 8 块 GPU 显存，不受 `--device cuda:N` 限制。**

- `--device cuda:N` 只控制 PyTorch 计算的 GPU，Isaac Sim 的 Omniverse Kit 渲染引擎会初始化所有可见 GPU
- p3_coarse 训练 (GPU 0-3) 本身是 1 个 Isaac Sim 实例
- 每个录制进程也是独立的 Isaac Sim 实例，会在全部 8 块 GPU 上分配渲染资源
- 多个 Isaac Sim 实例并发 → GPU 显存/渲染资源争抢 → 进程在渲染初始化阶段被静默杀死

### 验证过程

| 尝试 | 方式 | 结果 |
|------|------|------|
| `CUDA_VISIBLE_DEVICES=4` | 限制 GPU 可见性 | Isaac Sim 报 "Skipping NVIDIA GPU due CUDA being in bad state" |
| `--device cuda:4` (单录制) | 不设环境变量 | 卡在 `getBypassRenderSkelMeshProcessing` 后死亡 |
| `--device cuda:4/6/7` (3 并发) | 不同 GPU 分别录制 | 全部静默死亡 |
| `--device cuda:4` (单独录制, 清理后重试) | 仅有 1 个录制 + 训练 | 卡死 (0% CPU, sleeping)，12 分钟后确认失败 |
| p1_coarse (旧进程, 训练前) | `--device cuda:4` | 成功 (16 分钟出视频) — **唯一成功案例** |

Kit 日志无崩溃信息，dmesg 无 OOM 记录。进程直接消失或永久 sleeping。

### 最终结论

**Isaac Sim 录制与训练完全无法共存**，即使：
- 只运行 1 个录制进程
- 录制使用 idle GPU (cuda:4/6/7)
- 训练在另一组 GPU (cuda:0-3)

根本原因: Omniverse Kit 渲染管线会初始化**全部 8 块 GPU**（不受 `--device` 限制），与训练进程的 Isaac Sim 实例产生 GPU 渲染资源冲突，导致录制进程在渲染初始化阶段 (`getBypassRenderSkelMeshProcessing`) 卡死或死亡。

p1_coarse 的成功是唯一特例，可能与首次初始化时机、GPU 状态有关，不可复现。

**可行方案：等训练完全结束后再录制 Isaac Lab 视频，或使用 MuJoCo EGL 替代（不受 Isaac Sim 影响）。**

---
