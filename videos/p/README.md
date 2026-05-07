# Videos/p 目录说明

每个 sub-phase 文件夹内包含以下文件：

## 视频文件

| 文件 | 说明 |
|------|------|
| `*_isaaclab.mp4` | Isaac Lab 仿真录制（含速度箭头、地形可视化），200 帧 |
| `*_mujoco.mp4` | MuJoCo EGL 离屏渲染录制 |

## 参数文件 (params/)

每个文件夹内保留了该 phase 训练时的三个配置 yaml，用于追溯 checkpoint 对应的训练参数：

| 文件 | 记录内容 | 关键字段举例 |
|------|---------|-------------|
| **agent.yaml** | PPO 算法超参数与网络结构 | learning_rate, entropy_coef, gamma, lam, clip_param, actor/critic_hidden_dims, num_learning_epochs |
| **env.yaml** | 环境配置（仿真物理、地形、奖励函数、观测/动作空间、终止条件、速度命令范围、domain randomization） | reward weights, velocity ranges, terrain config, termination conditions, observation scales |
| **deploy.yaml** | 部署配置（PD 增益、action scale、默认关节角、关节映射、观测预处理） | stiffness, damping, action scale, default_joint_pos, joint_ids_map, observation history_length |

## velocity_env_cfg.py

部分文件夹内还有 `velocity_env_cfg.py`，是该 phase 生成的完整环境配置 Python 文件（env.yaml 的源码形式）。
