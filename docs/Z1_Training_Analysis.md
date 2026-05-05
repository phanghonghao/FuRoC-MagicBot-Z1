# Z1 Locomotion 训练分析

> 记录当前训练 run 的详细指标和 checkpoint 分析。
>
> 更新时间：2026-05-05

---

## 当前训练

| 项目 | 值 |
|------|-----|
| Run 名称 | s4_full_terrain |
| 起始 Checkpoint | model_5000.pt (from s3_rough_l1_4gpu) |
| 迭代进度 | ~5,400 / 50,000 |
| GPUs | 4x (cuda:0-3, torchrun, orchestrator 自动管理) |
| 训练开始时间 | 2026-05-04 16:56 |
| Orchestrator PID | 3664478 |
| 训练 PID | 3664479 (torchrun) |
| 核心改动 | 完整粗糙地形 (flat+random_grid+stairs+gap+boxes)，从 s3_rough_l1_4gpu best checkpoint 继续 |
| 自动化 | Training Orchestrator 管理，过拟合自动停止 |

### 关键配置

| 参数 | s3_rough_l1_4gpu (之前) | s4_full_terrain (当前) | 说明 |
|------|---------------|----------------------|------|
| 地形 | gentle terrain | rough terrain (全类型) | 完整粗糙地形配置 |
| action_rate 惩罚 | L1 | L1 | 继续使用 L1 |
| env_config | velocity_env_cfg_s3_rough_l1.py | velocity_env_cfg_s4_full_terrain.py | 新增完整地形 |
| 管理 | 手动 nohup | Orchestrator 自动 | 过拟合自动 kill+保存 |
| 监控阈值 | — | action_rate < -1.5 | 粗糙地形放宽 |

### 指标趋势

| 指标 | 早期 (iter ~5050) | 中期 (iter ~5155) | 最新 (iter ~5366) |
|------|-------------------|-------------------|-------------------|
| mean_reward | 6.46 | 11.93 | 16.97 |
| peak_reward | 8.00 | 37.73 | 37.73 |
| best_model | m5050 | m5155 | m5155 |
| status | HEALTHY | HEALTHY | HEALTHY |

### Orchestrator 状态

- 计划文件: `training_plans/z1_s4_s5_plan.yaml`
- 轮询间隔: 120s
- 日志: `/tmp/orchestrator_s4.log`
- 过拟合检测: 2000 iter warmup 后启动 (即 iter ~7000 后)

---

## 仿真视频记录

### s3_rough_l1 — model_3700

| Item | Value |
|------|-------|
| Date | 2026-05-04 |
| Checkpoint | model_3700.pt (iter 3700) |
| Video | `s3_rough_l1_model3700_isaaclab.mp4` |
| Save dir | `RTX6000/Magicbot_Z1/videos/s3_rough_l1/` |
| Platform | Spark IsaacLab (GB10, ARM) |
| Video length | 200 steps |
| Rollout time | 17.1s (12 FPS) |
| Action stats | mean_abs=0.86, range [-5.02, 4.70] |
| Percentiles | p50: 0.69, p90: 1.87, p99: 3.02 |
| Notes | L1 action rate penalty, gentle terrain |

---

## 历史分析

### s3_rough_l1_4gpu (Stage 3 尝试)

- Run dir: `2026-05-04_12-40-26_s6_l1_action_rate_4gpu`
- 最佳模型: model_5032 (reward: 31.2)
- 状态: OVERFITTING — reward 从 peak 33.28 下降 36%
- 迭代: 1700 → 7735 (4-GPU 分布式)
- 4卡显著提升了训练速度和 reward 上限 vs 单卡 (5.86 vs 31.2)

### s3_rough_l1 (Stage 3 单卡测试)

- Run dir: `2026-05-04_11-19-50_s6_l1_action_rate`
- 最佳模型: model_1778 (reward: 5.86)
- 状态: OVERFITTING — reward 从 peak 7.37 下降 34.2%
- 验证了 L1 action_rate 可行，但单卡 reward 上限低

### s2_gentle (Stage 2)

- 最佳模型: model_47862 (reward: 47.06)
- 状态: HEALTHY，完整跑完 50K iter
- 问题: model_49999 的 raw actions 均值 ~109，策略发散（L2 action_rate 崩塌）
- 教训: 不一定最后的模型最好，需要在 reward 峰值附近选择 checkpoint

### s3_rough_l2 (Stage 3)

- 最佳模型: model_32790 (reward: 38.04)
- 状态: OVERFITTING — action_rate 崩塌到 -167

### 训练演进图

```
s1_flat ──→ s1_flat_retry
  └──→ s3_rough_fail (失败) ──→ s2_gentle (✓, best=m47862)
                               └──→ s3_rough_l2 ──→ s3_rough_l1 (1卡, 过拟合)
                                                      └──→ s3_rough_l1_4gpu (4卡, 过拟合)
                                                                   └──→ s4_full_terrain (自动编排, 训练中)
```
