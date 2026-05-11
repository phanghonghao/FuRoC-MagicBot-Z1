# Training Log — 2026-05-12

## [01:34] Session Summary — p4_coarse 重启 + 23DOF 投篮任务规划

### Completed

| # | Item | Details |
|---|------|---------|
| 1 | p4_coarse 训练重启 | 从 p3_fine model_6900.pt 恢复，min_iterations 5000→2000 |
| 2 | Bug fix: orchestrator crash recovery 重置状态 | **Root cause**: orchestrator 发现死 PID → `_init_new_run()` → 清空 stage_history → 从 p1_coarse 重新开始 · **Fix**: 写入 `current_stage_status: "pending"` + `training_pid: null` + 填充 `stage_history` 6 条绝对路径 → 跳过 crash recovery 直接进 `_start_sub_phase()` · **Files**: RTX `orchestrator_state.json` |
| 3 | Bug fix: orchestrator 重启 3 次均失败 | **Root cause**: `--fresh` 清空状态、死 PID 触发 `_init_new_run()`、`--start-from` 不填充 stage_history · **Fix**: 手动构造完整 state JSON，用绝对 checkpoint 路径 · **Files**: RTX `orchestrator_state.json` |
| 4 | 确认 p3_fine > p3b_fine | p3_fine reward 33.33 (May 11) vs p3b_fine reward ~21 (May 9)，p3b 为旧 pipeline 残留 |
| 5 | 标记 p3b 为 DEPRECATED | 更新 `docs/tracking/bestmodel_phase.json`，p3b status 改为 "DEPRECATED — 待移除" |
| 6 | 确认 23DOF 投篮设计文档完整 | `docs/23dof_throwing/Z1_Throwing_Task_Plan.md` 包含完整实施计划（URDF、环境配置、奖励函数、PPO 参数） |

### Uncompleted / Blocked

| # | Item | Blocker | Next Step |
|---|------|---------|-----------|
| 1 | 23DOF 投篮训练启动 | 用户指出：当前策略导致两个关节在非平地扭矩不对称，需先加对称 reward | 确认对称 reward 加在哪个任务（locomotion 还是 throwing），添加后重新训练 |
| 2 | p4_coarse 仍在运行 | bad_ori 58%，用户判断会失败，决定跳过 rough terrain 直接转投篮 | 停掉 p4_coarse（orchestrator PID 1473792, training PID 1473864） |
| 3 | p3_fine → p4_coarse 地形跨度过大 | gentle → rough 直接跳，bad_ori 58% 证实用户预判 | 用户决定放弃 rough terrain 训练，转投篮任务 |

### Key Decisions

- p4_coarse (rough terrain) 放弃：gentle→rough 跨度过大，bad_ori 58%，训练无意义
- p3b 整个 phase 标记为待移除：p3_fine 在 gentle terrain 上 reward 33 优于 p3b 的 ~21
- 下一步转向 23DOF 投篮任务，但需先解决关节扭矩不对称问题
- orchestrator 重启的正确方式：手动写 state JSON（status="pending", pid=null, stage_history 填充绝对路径），不能依赖 --start-from 或 --fresh

## [18:00] Session Summary — 添加 joint_mirror 对称 reward + 启动 p3b_fine_symmetry 续训

### Completed

| # | Item | Details |
|---|------|---------|
| 1 | config_generator.py 新增 joint_mirror | 在 `_REWARD_DEFS` 字典添加 `joint_mirror` 条目，映射到 `mdp.joint_mirror`，6 对镜像关节 · **Files**: `magiclab_rl_lab/scripts/automation/config_generator.py` |
| 2 | z1_5phase_plan.yaml 新增 p3b_fine_symmetry | 在 p3b phase 的 sub_phases 末尾添加 `p3b_fine_symmetry` 子阶段：max_iter=5000, LR=1e-4, joint_mirror weight=-0.5, terrain=flat 70% + random_grid 30% (difficulty 0.35) · **Files**: `magiclab_rl_lab/training_plans/z1_5phase_plan.yaml` |
| 3 | Z1_5Stage_Training_Plan.md 新增 §9.5 | 记录左右关节不对称问题（hip_pitch 0.37rad, hip_yaw 0.52rad）、原因、方案 · 重编号 §9.5→§9.6, §9.6→§9.7 · **Files**: `docs/plans/Z1_5Stage_Training_Plan.md` |
| 4 | YAML + config_generator 本地验证通过 | YAML syntax OK, `joint_mirror` in `_REWARD_DEFS` OK |
| 5 | 修改文件同步到 RTX | `scp config_generator.py` + `scp z1_5phase_plan.yaml` 上传到 RTX |
| 6 | Kill p4_coarse + orchestrator | 停止 PID 1473863 (orchestrator) + 1473864/1473933-1473936 (4 GPU workers) |
| 7 | Bug fix: orchestrator stage_history 被清空 | **Root cause**: orchestrator 检测旧 PID dead → `current_stage_status: "running"` → mark as "failed" → `_init_new_run()` 清空 stage_history → checkpoint=None → 从零开始训练 · **Fix**: 手动写 `orchestrator_state.json`（status="pending", pid=null, stage_history 含 p3_fine model_6900.pt 绝对路径）→ orchestrator 直接进入 main loop → `_resolve_checkpoint` 正确找到 p3_fine 的 checkpoint |
| 8 | p3b_fine_symmetry 成功启动 | `--resume --load_run=2026-05-11_08-33-26_p3_fine --checkpoint=model_6900.pt` 正确传入，4 GPU, 4096 envs, PID 1495560 (orchestrator) + 1495567 (training) |

### Uncompleted / Blocked

| # | Item | Blocker | Next Step |
|---|------|---------|-----------|
| 1 | 验证 joint_mirror reward 在 Isaac Sim 运行时生效 | 训练刚启动，Isaac Sim 初始化中（~5-10min） | 等 `--tail` 出现 `Episode_Reward/joint_mirror` 确认生效 |
| 2 | p3b_fine_symmetry 训练进行中 | 需等 5000 iter 完成 | `/gpu-train --tail` 监控进度 |

### Key Decisions

- 对称 reward 放在 p3b（而非 p3）下：p3b 为过渡 phase，p3_fine 的 model_6900.pt 是当前最优 checkpoint
- joint_mirror weight = -0.5：温和引入不破坏已有步态（L2 squared 平均后单对 ~0.05 级别 × -0.5 ≈ -0.025 偏置）
- LR = 1e-4：比 p3_fine (2e-4) 更低，微调不破坏
- 地形与 p3b_fine 一致但简化（flat 70% + random_grid 30% difficulty 0.35），保持同分布只加对称约束
- orchestrator 重启方式与早前 session 一致：手动写 state JSON，status="pending" 避开 PID dead recovery 路径
