# Training Log — 2026-05-13

## [00:32] Session Summary — 减少p3→p4地形跨度，平滑过渡方案

### Completed

| # | Item | Details |
|---|------|---------|
| 1 | 修改 p4 phase-level terrain | flat 50%, random_grid 25% [0,0.45], stairs 10% [0,0.25], gap 5% [0,0.15], boxes 5% [0,0.15] |
| 2 | 修改 p4_coarse rewards + PPO | LR→3e-4, track_lin→1.8, track_ang→0.85, alive→0.2, energy→-2e-5, feet_slide→-0.1, stand_still→-1.5, 新增 joint_mirror→-0.2, action_rate_threshold→-1.2 |
| 3 | 修改 p4_fine terrain override | flat 40%, random_grid 25% [0,0.55], stairs 15% [0,0.40], gap 10% [0,0.25], boxes 10% [0,0.25] |
| 4 | 修改 p4_fine rewards | alive→0.18, action_rate_l1→-0.05, energy→-3e-5, feet_slide→-0.2, stand_still→-2.5, joint_mirror→-0.3, undesired_contacts→-1.5, LR→3e-4 |
| 5 | 新增 p5_coarse terrain + rewards | terrain: flat 30%, random_grid 20% [0,0.70], stairs 20% [0,0.55], gap 10% [0,0.40], boxes 10% [0,0.40]; rewards: energy→-4e-5, feet_slide→-0.3, stand_still→-1.0, joint_mirror→-0.3 |
| 6 | SCP + dry-run 验证 | SCP 到 RTX，dry-run 确认 p4_coarse LR=3e-4, threshold=-1.2；共 13 sub-phases |
| 7 | 重启 orchestrator | Kill 旧 orchestrator PID 2147598，--adopt 新 orchestrator PID 2285563 |
| 8 | 验证 adopt 成功 | p3_fine iter=4624/15000, reward=28.66, peak=36.06, HEALTHY |

### Uncompleted / Blocked

| # | Item | Blocker | Next Step |
|---|------|---------|-----------|
| 1 | p3_fine 完成后自动进入 p4_coarse | 等待训练完成 | `/gpu-train --orchestrator --status` 监控 |
| 2 | 验证 p4_coarse reward 无断崖式下跌 | p3_fine 未完成 | p4_coarse 启动后观察前 1000 iter reward |

### Key Decisions

- p3→p4 地形从"悬崖式"跳变改为渐进式：p4_coarse 仅引入 20% 新地形(stairs+gap+boxes)，difficulty 上限 0.25/0.15/0.15
- p4_coarse 保留较高速度跟踪奖励 (track_lin=1.8)，因为地形仍偏简单
- p4→p5 进一步渐进：p5_coarse 提升到 stairs 20% [0,0.55]，为 p5_fine 最终目标做铺垫
- joint_mirror 奖励从 p3b_fine_symmetry 开始引入，p4/p5 持续保留

### Parameter Changes Summary (p3_fine → p4_coarse → p5_fine)

```
           flat%  grid%  grid_diff  stairs%  stair_diff  gap%  gap_diff  box%  box_diff
p3_fine:    70     30    [0,0.35]     -         -        -      -        -      -
p4_coarse:  50     25    [0,0.45]    10       [0,0.25]   5    [0,0.15]   5    [0,0.15]
p4_fine:    40     25    [0,0.55]    15       [0,0.40]  10    [0,0.25]  10    [0,0.25]
p5_coarse:  30     20    [0,0.70]    20       [0,0.55]  10    [0,0.40]  10    [0,0.40]
p5_fine:    20     20    [0,0.70]    20       [0,0.60]  20    [0,0.50]  20    [0,0.50]  ← 最终目标
```

### Files Changed

- `magiclab_rl_lab/training_plans/z1_5phase_plan.yaml` — p4 terrain, p4_coarse/fine rewards, p5_coarse terrain+rewards
- RTX: `~/magiclab_rl_lab/training_plans/z1_5phase_plan.yaml` — synced via SCP
