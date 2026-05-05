# Z1 12DOF 学习曲线分析

> 更新时间: 2026-05-05
> 数据来源: TensorBoard event files + `best_models.json`
> 训练硬件: 4× RTX 6000D (85.7 GB), Isaac Lab 0.47.2, Isaac Sim 4.5.0
> 生成脚本: `scripts/plot_learning_curves.py`

---

## 概览

共 12 个有效 run（排除多 GPU 测试和重复启动）。按 best model reward 降序排列：

| Run | 别名 | 迭代数 | Peak Reward | Best Model | Best Iter | 状态 |
|-----|------|--------|-------------|------------|-----------|------|
| s1_flat | `s1_flat` | 36,429 | 48.35 | model_3861 (47.33) | 3,861 | OVERFITTING |
| s1_flat_retry | `s1_flat_retry` | 31,173 | 48.35 | model_3861 (47.33) | 3,861 | OVERFITTING |
| s2_gentle | `s2_gentle` | 49,999 | 48.10 | model_47862 (47.06) | 47,862 | HEALTHY |
| s3_rough_l2 | `s3_rough_l2` | 41,590 | 38.94 | model_32790 (38.04) | 32,790 | OVERFITTING |
| s3_rough_l1_4gpu | `s3_rough_l1_4gpu` | 7,735 | 33.28 | model_5032 (31.20) | 5,032 | OVERFITTING |
| s1_highspeed | `s1_highspeed` | 3,812 | 30.87 | model_2997 (30.11) | 2,997 | OVERFITTING |
| s1_stable | `s1_stable` | 12,834 | 29.99 | model_1555 (28.93) | 1,555 | OVERFITTING |
| s3_rough_l1 | `s3_rough_l1` | 1,799 | 7.37 | model_1778 (5.86) | 1,778 | OVERFITTING |
| s3_rough_fail | `s3_rough_fail` | 3,143 | 2.86 | model_1933 (1.85) | 1,933 | OVERFITTING |
| s3_rough_l1_mgpu | `s3_rough_l1_mgpu` | 50,018 | -0.46 | model_49999 (-0.46) | 49,999 | OVERFITTING |
| s3_rough_l1_mgpu_4gpu | `s3_rough_l1_mgpu_4gpu` | 50,070 | -1.68 | model_49999 (-1.68) | 49,999 | OVERFITTING |
| s4_full | `s4_full` | **训练中** | TBD | TBD | — | TRAINING |

### 状态说明

- **HEALTHY**: 训练正常，reward 仍在上升或维持高位
- **OVERFITTING**: reward 从 peak 大幅下降（>20%）或 action_rate 异常
- **TRAINING**: 仍在训练中，数据不完整

---

## Plot 1: Reward 对比分析

![Reward Comparison](1_reward_comparison.png)

### 分析

**表现最好的 runs:**
- **s1_flat / s1_flat_retry** (peak ~48.3): 最高 reward，但均在 ~3k-4k iter 后崩溃。s1_flat 是最初的 flat terrain 基准，reward 从 peak 48.35 骤降到 -0.96（下降 103%）
- **s2_gentle** (peak ~48.1, best 47.06@47862): **唯一健康完成的 run**。使用温和地形课程，50k iter 仍保持 ~45 reward，收敛稳定

**收敛速度对比:**
- s1_flat / s1_flat_retry: ~2.7k iter 达到 peak（最快），但随后崩溃
- s2_gentle: ~48k iter 达到 peak（最慢但最稳）
- s1_highspeed / s1_stable: ~1.5k-3k iter 达到 peak，但后续 reward 急剧下降

**过拟合模式:**
- 早期 runs (s1 阶段) 在达到 peak 后 reward 迅速下降到负值，表现为典型的 policy collapse
- s3_rough_l2 在 32.8k iter 达到 peak 38.9 后也崩溃
- 多 GPU runs (s3_rough_l1_mgpu, s3_rough_l1_mgpu_4gpu) 始终无法学到有效策略，reward 为负

---

## Plot 2: Reward 分解 (Focus: s4_full)

![Reward Decomposition](2_reward_decomposition_s4_full.png)

### 分析

**主要奖励组件贡献（按重要性排序）**

| 组件 | 贡献 | 说明 |
|------|------|------|
| +tracking_lin_vel | 正向主导 | 线速度跟踪奖励，是最主要的正向驱动力 |
| +tracking_ang_vel | 正向 | 角速度跟踪 |
| +alive | 正向（小） | 存活奖励 |
| -action_rate | 负向 | 动作平滑性惩罚，抑制抖动 |
| -torques | 负向（小） | 关节力矩惩罚 |
| -dof_vel | 负向（小） | 关节速度惩罚 |

**Curriculum 进度影响:**
- Terrain level 随训练逐步上升（从 0 到更高等级）
- Velocity cmd level 同步增加，任务难度逐步提升
- Curriculum 的渐进式增加使得 policy 能逐步适应更复杂的地形和速度指令

---

## Plot 3: 终止原因分析 (Focus: s4_full)

![Termination Analysis](3_termination_s4_full.png)

### 分析

**终止原因变化趋势:**
- `bad_orientation`（红色）: 训练初期占主导（接近 100%），随着训练进行逐渐下降。说明 robot 初始频繁摔倒
- `time_out`（绿色）: 随训练进行逐步上升，表示 episode 能完整执行到超时（20s）。这是训练进步的标志
- `base_height`（橙色）: 偶发，表示 base 高度超出范围

**Episode 长度变化:**
- 训练初期 episode 很短（robot 立即摔倒）
- 随着训练进行，episode 逐渐接近最大长度（1000 steps = 20s @ 50Hz）
- Episode 长度增长与 `time_out` 比例上升直接相关

**理想模式:** 训练后期 `time_out` 占 80%+，`bad_orientation` 低于 10%

---

## Plot 4: 训练效率 (Focus: s4_full)

![Training Efficiency](4_efficiency_s4_full.png)

### 分析

**Throughput (FPS):**
- 单 GPU: ~176k steps/s (4096 envs)
- 4 GPU: ~524k steps/s (32768 envs)，3× 吞吐提升（但 8× envs）
- Throughput 非 线性 增长，PPO 同步开销是瓶颈

**Collection vs Learning Time:**
- Collection time: 仿真数据收集耗时，通常占总时间的主要部分
- Learning time: PPO 策略更新耗时
- 随 envs 增加，collection time 增长较慢（并行仿真），learning time 增长较快（更大的 batch）

**Entropy (探索度):**
- 初始高（~4.0+），表示策略高度随机
- 随训练逐步下降，表示策略逐渐确定
- 如果 entropy 过早降到 0，说明探索不足，可能过拟合

**Learning Rate Schedule:**
- 使用 adaptive schedule（rsl-rl 默认），随 KL divergence 自适应调整
- 通常在训练后期逐步衰减

---

## 关键发现与建议

### 跨 Run 比较总结

1. **Flat terrain 训练不稳定**: s1_flat 和 s1_flat_retry 都在达到 ~48 reward 后崩溃，flat terrain 缺乏多样性导致后期过拟合
2. **温和地形课程最有效**: s2_gentle 使用渐进式地形课程，在 50k iter 达到 peak 47.06 且保持稳定，是唯一健康完成的 run
3. **Rough terrain 过早崩溃**: s3_rough_l2 在 32.8k iter 达到 peak 38.9 后崩溃，可能 terrain 难度上升过快
4. **L1 action rate 惩罚效果有限**: s3_rough_l1 加入 action rate L1 惩罚，但 peak 仅 7.37，效果不明显
5. **多 GPU 未改善训练质量**: s3_rough_l1_mgpu 和 s3_rough_l1_mgpu_4gpu 使用 4 GPU 训练但 reward 为负，可能是配置问题

### 最佳 Checkpoint 推荐

| 用途 | 推荐 Checkpoint | 备注 |
|------|-----------------|------|
| **部署/仿真验证** | s2_gentle / model_47862 (47.06) | 最稳定，50k iter |
| **Flat 基准对比** | s1_flat / model_3861 (47.33) | Reward 最高但需注意崩溃 |
| **Rough terrain** | s3_rough_l2 / model_32790 (38.04) | 唯一 rough terrain 数据点 |

### 下一步训练建议

1. **继续 s4_full 训练**: 当前正在使用 32768 envs / 4 GPU 训练，预计需要 ~55k iter
2. **调整 curriculum 节奏**: 参考 s4_gentle 的成功经验，放缓 terrain 难度提升速度
3. **监控 action_rate**: 保持 action_rate > -1.0（避免 policy collapse）
4. **Early stopping**: 如果 reward 连续 5k iter 下降 >20%，考虑停止
5. **Learning rate 调整**: 32768 envs 时考虑将 max_iterations 按比例缩小到 ~14k

---

## 生成命令

### 在 RTX 服务器上生成 plots

```bash
# SSH 到服务器
ssh phh@192.168.120.155

# 激活环境
cd ~/magiclab_rl_lab && source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab

# 生成全部 4 张 plot（自动选择 focus run）
python scripts/plot_learning_curves.py \
  --log_root logs/rsl_rl/magiclab_z1_12dof_velocity \
  --output_dir plots

# 指定 focus run
python scripts/plot_learning_curves.py \
  --log_root logs/rsl_rl/magiclab_z1_12dof_velocity \
  --output_dir plots \
  --focus_run 2026-05-04_16-56-05_s4_full_terrain
```

### 下载到本地

```bash
# 下载所有 PNG
scp phh@192.168.120.155:~/magiclab_rl_lab/plots/*.png \
  "D:/Desktop_Files/GPU-Train/RTX6000/Magicbot_Z1/plots/"
```

### 使用 /plot-train-Z1 Skill

```bash
# 生成全部 plots + 更新文档
/plot-train-Z1

# 指定 focus run
/plot-train-Z1 --focus s4_full

# 从服务器同步最新数据
/plot-train-Z1 --sync

# 只更新分析文档
/plot-train-Z1 --update-readme
```

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `1_reward_comparison.png` | 所有 run 的 mean reward 对比 |
| `2_reward_decomposition_s4_full.png` | s4_full 的 reward 分解 + curriculum |
| `3_termination_s4_full.png` | s4_full 的终止原因 + episode 长度 |
| `4_efficiency_s4_full.png` | s4_full 的训练效率诊断 |
| `README.md` | 本文档 |
