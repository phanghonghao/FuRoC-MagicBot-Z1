# Z1 12DOF 训练指标参考

> 训练指标分两大类：跨 run 可比（行为指标）和仅同 run 内可比（训练指标）。
> 判断 policy 好坏看行为指标，判断训练趋势看训练指标。

---

## 一、跨 Run 可比指标（行为指标）

这些指标不依赖 reward 权重设置，在不同 run、不同 terrain 配置之间可以直接比较。**用于评价 policy 质量和选择最佳 checkpoint。**

| 指标 | TensorBoard Tag | 范围 | 方向 | 理想值 | 说明 |
|------|----------------|------|------|--------|------|
| **time_out** | `Episode_Termination/time_out` | 0~1 | 越高越好 | >0.8 | Episode 成功完成（走满 20s）的比例 |
| **ep_len** | `Train/mean_episode_length` | 0~1000 | 越高越好 | >950 | 平均 episode 步数，1000 = 走满 |
| **bad_ori** | `Episode_Termination/bad_orientation` | 0~1 | 越低越好 | <0.1 | 机器人摔倒（姿态异常）的比例 |
| **vel_err** | `Metrics/base_velocity/error_vel_xy` | 0+ m/s | 越低越好 | <0.3 | 线速度跟踪误差，反映指令服从度 |
| **vel_yaw_err** | `Metrics/base_velocity/error_vel_yaw` | 0+ rad/s | 越低越好 | <0.5 | 偏航角速度跟踪误差 |

### 快速评估标准

| 等级 | time_out | ep_len | bad_ori | vel_err |
|------|----------|--------|---------|---------|
| 优秀 | >90% | >950 | <5% | <0.3 |
| 良好 | >70% | >800 | <15% | <0.5 |
| 一般 | >40% | >500 | <30% | <0.8 |
| 较差 | <40% | <500 | >30% | >0.8 |

### 用法

**选择最佳 checkpoint**: 在同一 run 内，挑 time_out 最高 + bad_ori 最低的。
**对比不同 run**: 直接比这几个数。比如 s4_gentle (time_out=90%) vs s5_rough (time_out=60%)，说明 gentle terrain 上的 policy 更稳定。

---

## 二、仅同 Run 内可比指标（训练指标）

这些指标的含义依赖于具体的 reward 权重和环境配置，**只在同一个 run 内看趋势**才有意义。跨 run 对比会产生误导。

| 指标 | TensorBoard Tag | 用途 | 说明 |
|------|----------------|------|------|
| **mean_reward** | `Train/mean_reward` | 监控训练趋势 | reward 权重不同则总分不可比。同一 run 内：上升=改善，下降=可能过拟合 |
| **action_rate** | `Episode_Reward/action_rate` | 检测 policy collapse | < -1.0 通常意味着动作剧烈抖动，policy 可能崩溃 |
| **entropy** | `Loss/entropy` | 监控探索度 | 从高到低是正常的。如果过早降到 0，说明探索不足 |
| **policy_std** | `Train/mean_std` | 动作确定性 | < 0.01 说明 policy 输出几乎确定，可能过拟合 |
| **value_loss** | `Loss/value_function` | 价值函数质量 | 突然飙升 = value function 发散 |
| **surrogate_loss** | `Loss/surrogate` | 策略更新幅度 | 正常应接近 0，大幅偏离说明策略更新不稳定 |
| **terrain_levels** | `Curriculum/terrain_levels` | 课程进度 | 当前 terrain 难度等级 |
| **vel_cmd_levels** | `Curriculum/lin_vel_cmd_levels` | 课程进度 | 速度指令难度等级 |
| **throughput** | `Perf/total_fps` | 训练效率 | steps/s，用于估算 ETA |
| **collection_time** | `Perf/collection_time` | 效率诊断 | 仿真数据收集耗时 |
| **learning_time** | `Perf/learning_time` | 效率诊断 | PPO 策略更新耗时 |

### 同 Run 内的训练趋势判断

| 趋势 | 信号 | 含义 |
|------|------|------|
| reward 上升 + time_out 上升 + bad_ori 下降 | 正常学习 | 继续训练 |
| reward 下降 + time_out 下降 + bad_ori 上升 | policy collapse | 考虑回滚到较早 checkpoint |
| reward 下降 + time_out 不变 + bad_ori 不变 | reward 权重变化 | 检查 curriculum 是否改变了难度 |
| entropy 骤降 + policy_std < 0.01 | 过拟合 | 降低学习率或增加 entropy bonus |
| value_loss 飙升 (>100) | value function 发散 | 学习率过大或数据异常 |

---

## 三、Reward 分解（参考用）

Reward 各子项的**绝对值不可跨 run 比**（因为权重不同），但**符号和相对贡献**可以辅助理解 policy 行为：

| 子项 | 类型 | 含义 | 跨 run 可比？ |
|------|------|------|-------------|
| track_lin_vel_xy | + | 线速度跟踪 | 权重不同不可比 |
| track_ang_vel_z | + | 角速度跟踪 | 权重不同不可比 |
| alive | + | 存活奖励 | 通常权重固定，弱可比 |
| action_rate | - | 动作平滑惩罚 | < -1.0 通常有问题（弱可比） |
| joint_acc | - | 关节加速度惩罚 | 权重不同不可比 |
| feet_clearance | + | 抬脚高度 | 权重不同不可比 |
| feet_contact_number | + | 足部接触数 | 权重不同不可比 |

**注意**: `action_rate < -1.0` 是一个跨 run 通用的**异常信号**（不管权重怎么设，-1.0 已经意味着动作幅度很大），可以用作 policy collapse 的早期预警。

---

## 四、视频标注指标

`label_video.py` 只标注跨 run 可比的行为指标：

```
Run: s4_gentle
Model: model_47862
time_out: 92.7%        ← 能走完吗
ep_len: 966/1000       ← 走了多少步
bad_ori: 7.3%          ← 摔倒率
vel_err: 0.41 m/s      ← 速度跟踪精度
```

不标注 reward、terrain、iteration，因为这些跨 run 不可比且容易误导。
