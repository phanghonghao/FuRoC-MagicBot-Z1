# Training Log — 2026-05-11

## [03:00] Session Summary — Z1 投篮 RL 任务框架搭建 + 验证

### Completed

| # | Item | Details |
|---|------|---------|
| 1 | 创建 `docs/23dof_throwing/Z1_Throwing_Task_Plan.md` | 完整实施计划文档，含设计概览、文件结构、奖励函数设计、手部替换指南 |
| 2 | URDF 手掌 collision | `MagicBotZ1_23dof.urdf` 的 `left_hand_palm_link` 和 `right_hand_palm_link` 各加了 sphere collision (radius=0.04)，带 TODO 标记 |
| 3 | 23DOF 机器人配置 `MAGICLAB_Z1_23DOF_CFG` | `assets/robots/magiclab.py` 新增 23DOF 配置：6 组 actuator（legs/feet/shoulders/elbows/wrists/waist）、手臂初始姿态、23 个 SDK 关节名 |
| 4 | 投篮任务框架 `tasks/throwing/` | 11 个 Python 文件：env config（含 `HAND_CONFIG` 参数化）、PPO config、7 个奖励函数、6 个观测函数、gym.register |
| 5 | Bug fix: `omni.log` 导入失败 | **Root cause**: `throwing/robots/__init__.py` 加了 `from isaaclab_tasks.utils import import_packages`，触发 isaaclab 导入链需要 `omni.log`（仅 Omniverse 运行时可用） · **Fix**: 改为空文件（与 locomotion/robots 一致），让 `list_envs.py` 的 `_walk_packages` 递归发现 `z1/__init__.py` 中的 `gym.register` · **Files**: `tasks/throwing/robots/__init__.py` |
| 6 | Bug fix: `list_envs.py` 缺少 throwing | **Root cause**: `list_envs.py` 只遍历 `locomotion.robots`，不包含 `throwing.robots` · **Fix**: 添加 `"throwing.robots"` 到遍历列表 · **Files**: `scripts/list_envs.py` |
| 7 | RTX 服务器验证通过 | `isaaclab.sh -p train.py --task Magiclab-Z1-23dof-Throwing --num_envs 64 --headless --max_iterations 3` 成功运行 3 次迭代，~1000 steps/s，所有奖励/观测函数正常 |

### 未完成/待调参

| # | Item | Blocker | 下一步 |
|---|------|---------|--------|
| 1 | 机器人立刻摔倒 | `bad_orientation: 99.2%` — 初始姿态不适合投篮任务，`limit_angle=1.0` 可能过严 | 调整初始关节角度、放宽 termination 阈值、考虑锁定腿部关节 |
| 2 | 球从未在手掌上 | `ball_on_palm: 0.0` — 球初始位置可能偏离手掌，需要精确设置 reset 时的球位置 | 在 EventCfg 中添加 ball reset 事件，将球预置到 `left_hand_palm_link` 上方 |
| 3 | `quat_rotate_inverse` 弃用警告 | 大量 warning（不影响功能） | `observations.py` 中已有 `try/except` 兼容，可后续统一改为 `quat_apply_inverse` |
| 4 | 正式训练未开始 | 环境框架已就绪，但初始状态需要调参才能有效训练 | 按上述 1-2 修复后启动正式训练 |

### 关键设计决策

- 手部参数化：`HAND_CONFIG` 字典控制 `hand_type`/`ee_link`/`hand_joint_names`，换 URDF 只改配置
- 动作空间：手臂 10DOF + 腰 1DOF = 11 DOF（`fixed_palm` 模式），`dexterous` 模式自动扩展
- 投篮任务与 12DOF 行走任务完全独立，互不影响
- 文档独立文件夹 `docs/23dof_throwing/`，与 12DOF 行走文档分离

### 新增/修改文件清单

```
# 新增
docs/23dof_throwing/Z1_Throwing_Task_Plan.md
tasks/throwing/__init__.py
tasks/throwing/agents/__init__.py
tasks/throwing/agents/rsl_rl_ppo_cfg.py
tasks/throwing/mdp/__init__.py
tasks/throwing/mdp/rewards.py
tasks/throwing/mdp/observations.py
tasks/throwing/mdp/commands.py
tasks/throwing/mdp/curriculums.py
tasks/throwing/robots/__init__.py        (空文件)
tasks/throwing/robots/z1/__init__.py      (gym.register)
tasks/throwing/robots/z1/shoot_env_cfg.py

# 修改
assets/robots/magiclab.py                 (+MAGICLAB_Z1_23DOF_CFG)
data/robots/magicbot-Z1/urdf/MagicBotZ1_23dof.urdf  (手掌 collision)
scripts/list_envs.py                      (+throwing.robots)

# 删除
docs/Z1_Throwing_Task_Plan.md             (迁入 23dof_throwing/)
```
