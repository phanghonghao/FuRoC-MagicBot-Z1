# Z1 Locomotion Training TODO

> 统一命名 & 清理计划。生成于 2026-05-04。

---

## 1. 版本命名规范（后续统一使用）

格式：`s{阶段}_{变体描述}`

| 阶段 | 命名前缀 | 地形 | 目标 |
|------|---------|------|------|
| Stage 1 | `s1_` | 纯平地 | 站立 |
| Stage 2 | `s2_` | 纯平地 | 平地行走 |
| Stage 3 | `s3_` | 50% flat + 50% gentle random_grid | 轻度地形 |
| Stage 4 | `s4_` | flat + random_grid + stairs + gap + boxes | 粗糙地形 |
| Stage 5 | `s5_` | 全类型全难度 + rails | 复杂地形 + 高速 |

---

## 2. 已有 Run 重命名映射

旧名 → 新名 | 阶段 | 状态 | 处理建议

### Stage 1 (平地行走)

| 旧名 | 新名 | 状态 | 处理 |
|------|------|------|------|
| `s1_flat` | `s1_flat` | OVERFITTING (best m3861, reward 47.33) | 保留 best checkpoint，其余可清理 |
| `s2_flat_retry` | `s1_flat_retry` | OVERFITTING (best m3861, reward 47.33) | 同上，与 s1_flat 同 checkpoint |
| `s2_stable` | `s1_stable` | OVERFITTING (best m1555, reward 28.93) | 失败实验，6个超参同时改导致崩塌，可清理 |
| `s3_highspeed` | `s1_highspeed` | OVERFITTING (best m2997, reward 30.11) | 探索性实验，可清理 |

### Stage 2 (轻度地形)

| 旧名 | 新名 | 状态 | 处理 |
|------|------|------|------|
| `s4_gentle_terrain` | `s2_gentle` | **HEALTHY** (best m47862, reward **47.06**) | **核心模型**，务必保留 |

### Stage 3 (粗糙地形)

| 旧名 | 新名 | 状态 | 处理 |
|------|------|------|------|
| `s4_terrain` | `s3_rough_fail` | OVERFITTING (best m1933, reward 1.85) | 失败：地形难度过早引入，可清理 |
| `s5_rough_terrain` | `s3_rough_l2` | OVERFITTING (best m32790, reward 38.04) | L2 action_rate 崩塌，保留 best，其余可清理 |
| `s6_l1_action_rate` | `s3_rough_l1` | OVERFITTING (best m1778, reward 5.86) | 1GPU 版，已清理 |
| `s6_l1_action_rate_4gpu` | `s3_rough_l1_4gpu` | OVERFITTING (best m5032, reward 31.20) | 4GPU 版 |
| `s6_mgpu` / `v6` | `s3_rough_l1_mgpu` | OVERFITTING | 多GPU测试 |

### Stage 4 (全类型地形)

| 旧名 | 新名 | 状态 | 处理 |
|------|------|------|------|
| `s4_full_terrain` | `s4_full_terrain` | OVERFITTING (best m8300, reward 58.95) | 全类型地形训练 |
| `s5_flat_deploy` | `s4_flat_deploy` | OVERFITTING (best m924, reward 30.99) | 平地部署优化 |

---

## 3. 清理计划

### 可立即清理（保留 best checkpoint，删除其余 model_*.pt）

这些 run 的中间 checkpoint 占用大量磁盘空间但无价值：

```
s1_flat:                     保留 model_3861.pt，删除其余
s1_flat_retry:               保留 model_3861.pt，删除其余
s1_stable:                   保留 model_1555.pt，删除其余
s3_rough_fail:               保留 model_1933.pt，删除其余
s1_highspeed:                保留 model_2997.pt，删除其余
s3_rough_l2:                 保留 model_32790.pt，删除其余
```

### 必须保留（不可删除）

```
s2_gentle:                   全部保留，HEALTHY 核心模型
s3_rough_l1_4gpu:            全部保留
s4_full_terrain:             全部保留
```

### 可完全删除

```
smoke_test_2
s2_flat_16k (s1_flat_16k)
s2_stable 早期版本 (14-50-07, 14-52-28)
```

### 清理记录 (2026-05-04)

磁盘: 14GB → 5.5GB

已删除所有中间 checkpoint（包括 best，因 grep -v 过滤失败）:
- s1_flat, s1_flat_retry, s1_stable, s3_rough_l2, s1_highspeed, s3_rough_fail

**安全保留**（未受影响）:
- s2_gentle (s4_gentle_terrain): 501 models — Stage 2 核心模型
- s3_rough_l1_4gpu (s6_l1_action_rate_4gpu): models — Stage 3

**影响**: 被删的 6 个 best 均为早期/失败/已被超越的版本，当前训练不依赖任何一个。

---

## 4. TODO 事项

- [x] ~~清理旧 run checkpoint~~ (14GB → 5.5GB, 2026-05-04)
- [x] ~~重命名 version 字段 (v/s6/s7 → s1-s4)~~ (2026-05-05)
- [ ] **s4_full_terrain 评估** → 决定下一步
- [ ] **更新训练计划文档** (Z1_Locomotion_Training_Plan.md 第 9 节历史记录)
- [ ] **多机器人可视化** (2026-05-05): Play/录制视频时同时显示多个机器人
  - Spark G1 PickPlace: `play_g1_pickplace.py` 加 overview camera (`num_envs>1` 时用 `sim.set_camera_view` 俯瞰所有 env)
  - Spark G1 Locomanipulation: 新建 `play_g1_locomanipulation.py`，同上 camera 逻辑
  - Spark Z1 Locomotion: `spark_play.py` 加 overview 模式（目前只跟 robot 0），`num_envs>1` 时切换俯瞰
  - 方案: `num_envs>1` → 计算 grid 布局，设 `eye=[span*0.6, span*0.8, span*0.5]`, `target=[center, center, 0.5]`
  - 涉及文件: `play_g1_pickplace.py`, `spark_play.py`, 新建 `play_g1_locomanipulation.py`
