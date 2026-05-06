# s2_gentle — Stage 2 轻度地形 (历史最佳)

> 训练日期: 2026-05-01
> 状态: HEALTHY — 完整 50K 收敛

## 模型信息

| 项目 | 值 |
|------|-----|
| Run Dir | `2026-05-01_04-50-05_s2_gentle` |
| Best Model | `model_47862.pt` (reward: 47.06) |
| Peak Reward | 48.10 @ iter 47857 |
| Final Iteration | 49,999 |
| Final Reward | 45.27 |
| Final action_rate | -0.203 |
| Final std | 0.365 |
| 本地 .pt | `s2_gentle_model_47800.pt` (7.0MB, iter 47800, 接近 best 47862) |
| 备份 | `s4_gentle_policy.pt` (actor-only, 1.1MB) |
| Checkpoint Path | `logs/rsl_rl/.../2026-05-01_04-50-05_s2_gentle/model_47862.pt` |

## 备注

历史最佳。50K iter 完整训练收敛。model_49999 的 raw actions 均值~109（L2 action_rate 崩塌），需要在 reward 峰值附近选 checkpoint。

## 奖励/惩罚权重 (代码实际值)

> 来源: `velocity_env_cfg.py` commit `e762473` (action_rate_l2 版本)
> 注: s1/s2/s3 使用同一套权重，仅地形和 num_envs 不同。

| 奖励项 | 权重 | 函数 |
|--------|------|------|
| track_lin_vel_xy | **1.0** | track_lin_vel_xy_yaw_frame_exp |
| track_ang_vel_z | **0.5** | track_ang_vel_z_exp |
| alive | **0.15** | is_alive |
| base_height | **-10.0** | base_height_l2 (target=0.7) |
| flat_orientation_l2 | **-5.0** | flat_orientation_l2 |
| dof_pos_limits | **-5.0** | joint_pos_limits |
| stand_still | **-3.5** | stand_still_joint_deviation_l1 |
| base_linear_velocity | **-2.0** | lin_vel_z_l2 |
| base_angular_velocity | **-0.05** | ang_vel_xy_l2 |
| joint_deviation_legs | **-0.7** | joint_deviation_l1 (hip_roll, hip_yaw) |
| action_rate | **-0.05** | action_rate_l2 |
| feet_slide | **-0.2** | feet_slide |
| undesired_contacts | **-1.0** | undesired_contacts (threshold=1) |
| feet_clearance | **1.0** | foot_clearance_reward (std=0.05, target=0.1) |
| feet_contact_number | **0.5** | feet_contact_number (period=0.6) |
| energy | **-2e-5** | energy |
| joint_vel | **-0.001** | joint_vel_l2 |
| joint_acc | **-2.5e-7** | joint_acc_l2 |

## 地形 & 环境

| 参数 | 值 |
|------|-----|
| 地形 | 50% flat + 50% random_grid |
| num_envs | 4096 |
| difficulty_range | (0.0, 1.0) |
| 速度指令 | ranges: [-0.1, 0.1], limit: [-0.5, 1.0] |
| 域随机化 - 摩擦 | static: (0.3, 1.0), dynamic: (0.3, 1.0) |
| 域随机化 - 骨盆质量 | (0.7, 1.3) |
| 域随机化 - 全身质量 | (0.7, 1.3) |
| 域随机化 - 推力 | vx/vy: ±0.5, interval 5s |

## PPO 超参数

| 参数 | 值 |
|------|-----|
| learning_rate | 1e-3 (adaptive) |
| gamma | 0.99 |
| lam | 0.95 |
| clip_param | 0.2 |
| entropy_coef | 0.01 |
| desired_kl | 0.01 |
| num_steps_per_env | 24 |
| num_mini_batches | 4 |
| num_learning_epochs | 5 |
| max_grad_norm | 1.0 |
| init_noise_std | 1.0 |

## 训练链

```
s1_flat (m3861, 47.33, flat)
  └──→ s2_gentle (m47862, 47.06, gentle) ← 历史最佳
```
