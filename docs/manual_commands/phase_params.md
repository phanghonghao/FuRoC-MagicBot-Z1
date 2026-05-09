# 各阶段参数速查

> 来源：`training_plans/z1_5phase_plan.yaml`

## 阶段总览

| 阶段 | 地形 | 子阶段 | max_iter |
|------|------|--------|----------|
| p1 | 平地 bootstrap | p1_coarse → p1_fine | 5K + 5K |
| p2 | 平地 velocity tracking | p2_coarse → p2_fine | 10K + 10K |
| p3 | gentle terrain | p3_coarse → p3_fine | 15K + 15K |
| **p3b** | **intermediate** | **p3b_coarse → p3b_fine** | **15K + 15K** |
| p4 | rough terrain | p4_coarse → p4_fine | 25K + 25K |
| p5 | full terrain + polish | p5_coarse → p5_fine | 30K + 30K |

## PPO 参数

| 子阶段 | learning_rate | entropy_coef |
|--------|--------------|-------------|
| p1_coarse | 1e-3 | 0.01 |
| p1_fine | 5e-4 | 0.008 |
| p2_coarse | 1e-3 | 0.01 |
| p2_fine | 5e-4 | 0.008 |
| p3_coarse | 5e-4 | 0.01 |
| p3_fine | 3e-4 | 0.008 |
| **p3b_coarse** | **1e-4** | **0.015** |
| **p3b_fine** | **8e-5** | **0.012** |
| p4_coarse | 1e-4 | 0.012 |
| p4_fine | 8e-5 | 0.008 |
| p5_coarse | 1e-4 | 0.01 |
| p5_fine | 5e-5 | 0.005 |

## 地形配置

### p3b — Intermediate Terrain

| 地形类型 | 比例 | 难度范围 |
|---------|------|---------|
| flat | 50% | — |
| random_grid | 30% | 0.0–0.5 |
| stairs | 10% | 0.0–0.3 |
| boxes | 10% | 0.0–0.3 |

p3b_fine 进阶：grid 0-0.6, stairs 0-0.4, boxes 0-0.4

### p4 — Rough Terrain

| 地形类型 | 比例 | 难度范围 |
|---------|------|---------|
| flat | 30% | — |
| random_grid | 30% | 0.0–0.5 |
| stairs | 20% | 0.0–0.4 |
| gap | 10% | 0.0–0.3 |
| boxes | 10% | 0.0–0.3 |

p4_fine 进阶：grid 0-0.7, stairs 0-0.6, gap 0-0.5

### p5 — Full Terrain

| 地形类型 | 比例 | 难度范围 |
|---------|------|---------|
| flat | 20% | — |
| random_grid | 20% | 0.0–0.7 |
| stairs | 20% | 0.0–0.6 |
| gap | 20% | 0.0–0.5 |
| boxes | 20% | 0.0–0.5 |

## Reward 权重变化趋势

coarse → fine 的一般规律：
- `track_lin_vel_xy`: 1.5 → 2.0（更强速度跟踪）
- `alive`: 0.5 → 0.15（减少存活奖励，逼迫学习）
- `base_height`: -5 → -10（更强高度惩罚）
- `flat_orientation_l2`: -3 → -7（更严格姿态）
- `action_rate_l1`: -0.04 → -0.06（更平滑动作）
- `stand_still`: 0 → -3.5（fine 阶段惩罚不动）
- `undesired_contacts`: 0 → -2.0（fine 阶段惩罚碰撞）
- `feet_slide`: 0 → -0.5（fine 阶段惩罚滑脚）
