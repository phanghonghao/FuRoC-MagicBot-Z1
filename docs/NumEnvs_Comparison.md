# num_envs 参数对比分析 (4096 / 16384 / 32768)

> 硬件: 4× RTX 6000D (85.7 GB each), Isaac Lab 0.47.2, Isaac Sim 4.5.0
> 任务: MagicBot Z1 12DOF Locomotion, s4_full_terrain
> 日期: 2026-05-05

## 核心公式

```
每 iteration 的 timesteps = num_envs_per_gpu × num_gpus × num_steps(=24)
总训练量                 = max_iterations × num_envs_per_gpu × num_gpus × num_steps
```

- `num_steps = 24` — 每次 PPO 迭代中每个 env 收集的步数（固定值）
- `num_gpus = 4` — 当前使用 4 块 GPU
- `num_envs_per_gpu` — 每块 GPU 的并行环境数（可调参数）

---

## 一览表

| 指标 | 4096 / GPU | 16384 / GPU | 32768 / GPU |
|------|-----------|-------------|-------------|
| **每 GPU envs** | 4,096 | 16,384 | 32,768 |
| **总并行 envs** | 16,384 | 65,536 | 131,072 |
| | | | |
| **每 iter timesteps** | 393,216 | 1,572,864 | 3,145,728 |
| **每 iter timesteps (万)** | 39.3 万 | 157.3 万 | 314.6 万 |
| | | | |
| **实测 iter time** | ~2.2 s | ~4.5 s* | ~6.0 s |
| **实测 throughput** | ~176k steps/s | ~330k steps/s* | ~524k steps/s |
| **throughput 倍数** | 1.0× | ~1.9× | 3.0× |
| | | | |
| **GPU 显存占用** | ~6-8 GB | ~12-15 GB* | ~20.7 GB |
| **GPU 显存利用率** | ~8% | ~15%* | ~24% |
| **GPU 剩余空间** | ~78 GB | ~71 GB* | ~65 GB |
| | | | |
| **55k iters 总 timesteps** | 216 亿 | 865 亿 | 1,730 亿 |
| **等效 timesteps 倍数** | 1× | 4× | 8× |
| | | | |
| **等量 timesteps 所需 iters** | 55,000 | 13,750 | 6,875 |
| **等量 timesteps 的 ETA** | ~34 h | ~17 h* | ~11.5 h |

> *16384 / GPU 的数据为估算值（基于 4096 和 32768 的实测数据插值）

---

## 关键发现

### 1. Throughput 不是线性增长

```
envs 增长:   4096 → 16384 (4×) → 32768 (8×)
throughput:  176k → ~330k  (1.9×) → 524k (3.0×)
```

8 倍的 envs 只带来 3 倍的 throughput。原因是 **PPO 的 collection phase 是串行瓶颈**：
- 每个 iteration 中，仿真并行度提高了，但 PPO 更新（梯度计算、NCCL 同步）的时间也增加了
- GPU 计算不是瓶颈，显存带宽和通信才是

### 2. 显存利用率仍然很低

```
32768 envs: 20.7 GB / 85.7 GB = 24%

剩余空间: 85.7 - 20.7 = 65 GB (76% 空闲)
```

RTX 6000D 的 85 GB 显存远未被充分利用。理论上可以继续增加 envs，但 throughput 的边际收益会递减。

### 3. max_iterations 必须随 envs 调整

如果目标是 **覆盖相同的总 timesteps**：

```
目标: 55,000 iters × 16,384 envs × 24 = 216 亿 timesteps

4096  / GPU → max_iterations = 55,000   (基准)
16384 / GPU → max_iterations = 13,750   (÷4)
32768 / GPU → max_iterations =  6,875   (÷8)
```

如果不调整 `max_iterations`，用 32768 envs 跑 55k iters 会覆盖 **8 倍** 的训练量，导致：
- 训练时间从 ~11h 变成 ~20h+（浪费算力）
- learning rate schedule 不匹配（衰减过慢）
- 总梯度更新次数过多（可能过拟合）

### 4. 训练质量差异

| 方面 | 4096 envs | 32768 envs |
|------|-----------|------------|
| 每次更新的数据量 | 小 (39万) | 大 (314万) |
| 梯度估计方差 | 较高 | 较低（更稳定） |
| 数据多样性 | 每次较少 | 每次更多样 |
| 收敛速度 | 每步进步小 | 每步进步大 |

更多 envs 意味着每次 PPO 更新基于更丰富的经验，**梯度估计更准确，训练更稳定**。

---

## 推荐配置

| 场景 | 推荐值 | 理由 |
|------|--------|------|
| 快速实验 / debug | 4096 / GPU | 启动快，显存占用低 |
| 正式训练 (4 GPU) | **16384 / GPU** | 性价比最高，显存充裕 |
| 追求最快收敛 | **32768 / GPU** | throughput 最高，需配合调整 max_iterations |
| 极限压测 | 65536 / GPU | 显存可能接近上限，收益递减严重 |

> **注意**: 使用 16384 或 32768 时，务必按比例缩小 `max_iterations`，否则训练量会远超预期。
