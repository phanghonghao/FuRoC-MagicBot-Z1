# Training Log — 2026-05-09

---

## [02:20] Orchestrator Pipeline 重启 + KVDB 修复

### 完成

| # | 项目 | 详情 |
|---|------|------|
| 1 | 添加 `_wait_for_kit_cleanup()` | `phase_orchestrator.py:556-584`，在子阶段之间等待 Kit 进程完全退出，防止 KVDB 锁冲突 |
| 2 | 在 `_advance()` 中调用清理 | `phase_orchestrator.py:600` |
| 3 | 上传到 RTX + 验证 import | scp + `python -c "from scripts.automation..."` OK |
| 4 | Dry-run 验证 | 12 个子阶段全部正确 |
| 5 | 首次启动 pipeline | PID 1244556, 从 p3_coarse_v2 `model_10500.pt` (reward 37.38) 启动 |

---

## [04:35] Pipeline 调试 + 僵尸进程清理

### Bug 修复

| Bug | 原因 | 修复 | 文件 |
|-----|------|------|------|
| `AttributeError: best_model_path` | `_label_videos()` 用了错属性名 | `best_model_path` → `best_checkpoint_path` | `phase_orchestrator.py:934` |
| `--fresh` 清空 history, checkpoint 解析失败 | `_resolve_checkpoint()` 找不到上一个阶段的 best model | 手写 `orchestrator_state.json` 注入 p3_coarse_v2 | RTX `orchestrator_state.json` |
| 旧 run 目录数据污染 | 上次失败的 `2026-05-08_15-55-41_p3b_coarse` 仍匹配搜索 | 归档为 `_archived_p3b_coarse_old` | RTX `logs/rsl_rl/` |
| run dir 4/8 min 超时 | Isaac Sim 4-GPU 初始化 10+ 分钟 | `max_attempts = 28 if multi-gpu else 16` | `phase_orchestrator.py:387` |
| 多个 orchestrator 实例冲突 | 旧 orchestrator 杀掉新 orchestrator 启动的训练 | 显式 kill 旧 PID | — |
| `.pyc` 缓存阻止代码更新 | `__pycache__/` 旧字节码优先加载 | 删除 `__pycache__/` | — |

### 僵尸进程清理

| 进程 | PID | 问题 | 处理 |
|------|-----|------|------|
| `train_monitor.py` | 198656 | 运行 24h+，浪费资源 | `kill -9` |
| `omni.telemetry.transmitter` | 1328209 | 占用 GPU 4-7 各 ~1.9G | `kill -9` |

### 重新启动 pipeline

- Orchestrator PID 1365094, Training PID 1365095 (4 GPUs, torchrun)
- Run dir: `2026-05-09_03-22-15_p3b_coarse`
- 起始 checkpoint: p3_coarse_v2 `model_10500.pt` (reward 37.38)

### 待完成

| # | 项目 | 阻塞原因 |
|---|------|---------|
| 1 | p3b_coarse 训练中 | 运行中，ETA ~8h |
| 2 | 验证 KVDB 修复 | 等 p3b_coarse → p3b_fine 转阶段 |
| 3 | 验证 `max_attempts=28` 生效 | `.pyc` 缓存可能阻止了更新 |
| 4 | 后续阶段 p3b_fine → p4 → p5 | 依赖 p3b_coarse 完成 |

---

## [21:20] Session Summary — Orchestrator 修复 + p3b_fine MuJoCo 录制

### Completed

| # | Item | Details |
|---|------|---------|
| 1 | Training tail 检查 (×3) | `--tail --rtx` 三次快照: iter 12642→13335→14332, reward 1.35→10.84→7.80 |
| 2 | Bug fix: orchestrator 监控卡住 | **Root cause**: orchestrator `training_run_dir` 指向旧 run (`2026-05-08_13-03-53_p4_coarse`)，新训练创建了新 run dir (`2026-05-09_09-50-46_p4_coarse`)，TensorBoard reader 无法找到新 events 文件 · **Fix**: kill 旧 orchestrator，用 python 修改 `orchestrator_state.json` 中 `training_run_dir` 指向新 run，重启 resume 模式 · **Files**: RTX `orchestrator_state.json` |
| 3 | p3b_fine MuJoCo 视频录制 | 4-GPU 4×4096 envs 并行录制, 500 步, 9 falls, 已下载 + label |
| 4 | p3b_fine params 下载 | `scp -r params/` 到 `videos/p/p3b_fine/params/` |

### Uncompleted / Blocked

| # | Item | Blocker | Next Step |
|---|------|---------|-----------|
| 1 | p4_coarse 训练进行中 | iter 14332/35800 (40%), ETA ~18h | 等 pipeline 自动推进 |
| 2 | MuJoCo 地形适配讨论 | MuJoCo MJCF 只有 `type="plane"`, 无法还原 Isaac Sim 粗糙地形 | 可在 MJCF 加 hfield 或用 Isaac Sim 验证 |

### Key Decisions

- Orchestrator 状态文件修复: 直接 JSON 编辑 `training_run_dir` 字段，避免重新跑整个 pipeline
- p3b_fine MuJoCo 录制: 9 falls 反映 sim-to-sim gap (粗糙地形策略在平地表现不佳)
- MuJoCo 地形: 确认为纯平面 `type="plane"`，非物理粗糙，适合验证平地策略

---

## [21:25] MuJoCo 地形适配方案 — 各阶段对照表

### Isaac Sim vs MuJoCo 地形对照

当前 MJCF (`MAGICBOTZ1.xml`) 只有一个 `<geom type="plane">`，完全平坦。
以下是根据 `z1_5phase_plan.yaml` 各阶段地形配置，MuJoCo 对应的设置方案：

| Phase | Isaac Sim 地形类型 | sub_terrains 组成 | MuJoCo 方案 |
|-------|-------------------|-------------------|-------------|
| **p1** | `plane` (无 generator) | 纯平地 | `<geom type="plane">` (当前默认，无需改动) |
| **p2** | `plane` (无 generator) | 纯平地 | `<geom type="plane">` (当前默认，无需改动) |
| **p3** | `generator` (gentle) | flat 70% + random_grid 30% (difficulty 0.0-0.5) | `<hfield>` 低幅起伏，振幅 ~5mm (vertical_scale=0.005) |
| **p3b** | `generator` (intermediate) | flat 50% + random_grid 30% (0.0-0.6) + stairs 10% (0.0-0.4) + boxes 10% (0.0-0.4) | `<hfield>` + 散布 `<box>` 台阶/障碍物 |
| **p4** | `generator` (rough) | flat 30% + random_grid 30% (0.0-0.7) + stairs 20% (0.0-0.6) + gap 10% (0.0-0.5) + boxes 10% (0.0-0.5) | `<hfield>` 较强起伏 + `<box>` 台阶 + 地面缺口 |
| **p5** | `generator` (full) | flat 20% + random_grid 20% (0.0-0.7) + stairs 20% (0.0-0.6) + gap 20% (0.0-0.5) + boxes 20% (0.0-0.5) | `<hfield>` 高起伏 + 密集 `<box>` + 多段缺口 |

### MuJoCo 地形实现方式

#### 1. Flat (p1, p2) — 当前已有

```xml
<!-- 当前 MAGICBOTZ1.xml 默认 -->
<geom name="ground" type="plane" pos="0 0 0" friction="1 1 5"
      size="10 10 1" conaffinity="1" contype="1" material="MatGnd"/>
```

#### 2. Gentle Bumps (p3) — hfield 低幅起伏

```xml
<asset>
  <!-- 9x21 网格，匹配 Isaac Sim 的 num_rows=9, num_cols=21 -->
  <!-- size: x_half=4.0 y_half=4.0 base_z=0 max_elevation=0.01 (约2倍 vertical_scale) -->
  <hfield name="gentle_terrain" nrow="9" ncol="21"
          size="4 4 0 0.01" file="gentle_terrain.png"/>
  <!-- 需要生成 heightmap PNG: 白色=最高(0.01m), 黑色=最低(0m) -->
  <!-- 可从 Isaac Sim 导出 heightmap 数据 -->
</asset>

<worldbody>
  <!-- 替换原来的 plane -->
  <geom name="ground" type="hfield" hfield="gentle_terrain"
        pos="0 0 0" friction="1 1 5" conaffinity="1" contype="1"/>
</worldbody>
```

**p3 地形参数映射：**
- `horizontal_scale: 0.1` → MuJoCo 每个 hfield 格子 ~0.1m
- `vertical_scale: 0.005` → 最大高度变化 ~5mm（gentle）
- `difficulty_range: [0.0, 0.5]` → hfield 振幅约 0-2.5mm

#### 3. Intermediate (p3b) — hfield + 障碍物

```xml
<asset>
  <hfield name="intermediate_terrain" nrow="9" ncol="21"
          size="4 4 0 0.015" file="intermediate_terrain.png"/>
</asset>

<worldbody>
  <geom name="ground" type="hfield" hfield="intermediate_terrain"
        pos="0 0 0" friction="1 1 5" conaffinity="1" contype="1"/>

  <!-- Stairs: 台阶 (p3b stairs 10%, difficulty 0.0-0.4) -->
  <geom name="stair1" type="box" size="0.5 0.5 0.02" pos="2 0 0.02"
        friction="1 1 5" conaffinity="1" contype="1" rgba="0.6 0.5 0.4 1"/>
  <geom name="stair2" type="box" size="0.5 0.5 0.04" pos="2 1 0.04"
        friction="1 1 5" conaffinity="1" contype="1" rgba="0.6 0.5 0.4 1"/>

  <!-- Boxes: 障碍方块 (p3b boxes 10%, difficulty 0.0-0.4) -->
  <geom name="box1" type="box" size="0.1 0.1 0.03" pos="-1 1.5 0.03"
        friction="1 1 5" conaffinity="1" contype="1" rgba="0.5 0.5 0.5 1"/>
</worldbody>
```

#### 4. Rough (p4) — 强 hfield + 台阶 + 缺口

```xml
<asset>
  <hfield name="rough_terrain" nrow="9" ncol="21"
          size="4 4 0 0.025" file="rough_terrain.png"/>
</asset>

<worldbody>
  <geom name="ground" type="hfield" hfield="rough_terrain"
        pos="0 0 0" friction="1 1 5" conaffinity="1" contype="1"/>

  <!-- Stairs: 更多更高 (p4 stairs 20%, difficulty 0.0-0.6) -->
  <geom name="stair1" type="box" size="0.5 0.5 0.03" pos="2 0 0.03" .../>
  <geom name="stair2" type="box" size="0.5 0.5 0.06" pos="2 1 0.06" .../>
  <geom name="stair3" type="box" size="0.5 0.5 0.09" pos="2 2 0.09" .../>

  <!-- Gap: 地面缺口 (p4 gap 10%, difficulty 0.0-0.5) -->
  <!-- 用多段 plane 实现，中间留空 -->
  <geom name="ground_a" type="plane" pos="-3 0 -0.001"
        size="2 5 0.1" friction="1 1 5" conaffinity="1" contype="1"/>
  <!-- gap here: x ∈ [-1, -0.5] -->
  <geom name="ground_b" type="plane" pos="2 0 -0.001"
        size="2 5 0.1" friction="1 1 5" conaffinity="1" contype="1"/>
</worldbody>
```

#### 5. Full (p5) — 最复杂地形

```xml
<asset>
  <hfield name="full_terrain" nrow="9" ncol="21"
          size="4 4 0 0.035" file="full_terrain.png"/>
</asset>

<worldbody>
  <geom name="ground" type="hfield" hfield="full_terrain" .../>

  <!-- 密集障碍物: stairs 20% + boxes 20% + gaps 20% -->
  <!-- 需要程序化生成大量 geom -->
</worldbody>
```

### 关键参数对照

| 参数 | Isaac Sim | MuJoCo 对应 |
|------|-----------|-------------|
| `horizontal_scale: 0.1` | 每格 0.1m | hfield `nrow×ncol` 覆盖 `size` 区域 |
| `vertical_scale: 0.005` | 最大高度 5mm | hfield 第5个参数 `max_elevation` |
| `difficulty_range` | 控制地形复杂度 | 影响 hfield 振幅和障碍物密度 |
| `proportion` | 每种地形占比 | 通过 hfield 数据 + 额外 geom 比例模拟 |

### Heightmap 生成方案

hfield 数据需要从 Isaac Sim 导出或用 Python 程序化生成：

```python
# 方案 A: 从 Isaac Sim 导出
# 在 Isaac Sim 中生成地形后，导出 heightmap 为 PNG
# MuJoCo hfield 支持 PNG 格式 (灰度图)

# 方案 B: Python 程序化生成
import numpy as np
from PIL import Image

nrow, ncol = 9, 21
difficulty = 0.5  # 对应 p3_fine 的 max difficulty

# 随机起伏
hmap = np.random.rand(nrow, ncol) * difficulty
# 平滑处理 (模拟 gentle terrain)
from scipy.ndimage import gaussian_filter
hmap = gaussian_filter(hmap, sigma=1.5)

# 归一化到 0-255
hmap_img = (hmap / hmap.max() * 255).astype(np.uint8)
Image.fromarray(hmap_img).save("gentle_terrain.png")
```

### 总结

| Phase | MuJoCo 需要改动? | 改动量 | 优先级 |
|-------|-----------------|--------|--------|
| p1 | 不需要 | — | — |
| p2 | 不需要 | — | — |
| p3 | 需要 hfield | 小 (仅替换 ground geom) | 中 |
| p3b | 需要 hfield + 几个 box | 中 | 中 |
| p4 | 需要 hfield + 多个 box + gap | 大 | 高 |
| p5 | 需要 hfield + 大量 box + gap | 大 | 高 |
