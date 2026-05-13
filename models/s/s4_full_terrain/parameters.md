# s4_full_terrain — Stage 4 全类型粗糙地形

> 训练日期: 2026-05-04
> 状态: OVERFITTING — 训练停止在 15000，最佳在 8100

## 模型信息

| 项目 | 值 |
|------|-----|
| Run Dir | `2026-05-04_16-56-05_s4_full_terrain` |
| Best Model | `model_8100.pt` (reward: 67.64) |
| Peak Reward | 67.64 @ iter 8164 |
| Final Iteration | 15,000 |
| Final Reward | ~22.00 (overfitted) |
| 本地 .pt | `s4_full_terrain_model_8100.pt` (6.8MB) |
| 本地 JIT | `s4_full_terrain_policy.pt` (1.1MB) |
| Checkpoint Path | `logs/rsl_rl/.../2026-05-04_16-56-05_s4_full_terrain/model_8100.pt` |
| Resume From | `s3_rough_l1_4gpu/model_5000.pt` |

## 录制视频

| 视频 | 文件 | 大小 |
|------|------|------|
| Isaac Sim | `s4_full_terrain_m8100_isaaclab.mp4` | 20 MB |
| MuJoCo sim2sim | `s4_full_terrain_m8100_mujoco.mp4` | 444 KB |

## 备注

4-GPU orchestrator managed, rough terrain (全类型)。过拟合检测: 2000 iter warmup 后启动。

## 奖励/惩罚权重 (代码实际值)

> 来源: `velocity_env_cfg.py` commit `6ffbe57` (action_rate 改为 L1)
> 注: 与 s1/s2/s3 唯一区别是 action_rate 从 L2 改为 L1，权重不变。num_envs 本地改为 16384。

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
| action_rate | **-0.05** | action_rate_l1 ← **唯一变更** |
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
| 地形 | 20% flat + 30% random_grid + 20% stairs + 15% gap + 15% boxes |
| num_envs | **16384** (本地未提交修改) |
| difficulty_range | (0.0, 1.0) |
| 速度指令 | ranges: [-0.1, 0.1], limit: [-0.5, 1.0] |
| 域随机化 - 摩擦 | static: (0.3, 1.0), dynamic: (0.3, 1.0) |
| 域随机化 - 骨盆质量 | (0.7, 1.3) |
| 域随机化 - 全身质量 | (0.7, 1.3) |
| 域随机化 - 推力 | vx/vy: ±0.5, interval 5s |
| env_config | `velocity_env_cfg_s4_full_terrain.py` |

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

## 与前阶段的差异

| 参数 | s1/s2/s3 | s4_full_terrain |
|------|----------|-----------------|
| action_rate 函数 | action_rate_l2 | **action_rate_l1** |
| num_envs | 4096 | **16384** |
| 地形 | flat / gentle | 全类型混合 |

## 训练链

```
s1_flat (m3861, 47.33, flat)
  └──→ s2_gentle (m47862, 47.06, gentle)
         └──→ s3_rough_l2 (m32790, 38.04, rough)
                └──→ s4_full_terrain (m8100, 67.64, full terrain) ← 最佳 @8100, overfitted after 9000
```
