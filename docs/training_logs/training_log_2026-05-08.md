# Training Log — 2026-05-08

---

## [16:55] Session Summary — p3_coarse_v2 训练监控 + Orchestrator 文档编写

### Completed

| # | Item | Details |
|---|------|---------|
| 1 | p3_coarse_v2 训练 plots 生成 | 添加 `p3_coarse_v2` alias 到 plot script，生成 4 张 plots + PDF report。**Files**: `plots/p3_coarse_v2/` |
| 2 | p3_coarse_old_v1 训练 plots 生成 | 添加 `p3_coarse_old_v1` alias，生成旧版 v1 的 4 张 plots + PDF report。**Files**: `plots/p3_coarse_old_v1/` |
| 3 | best_models.json 更新 | RTX 上运行 train_monitor --once，下载更新后的 best_models.json |
| 4 | Orchestrator 运维文档编写 | 新建 `Z1_Orchestrator_Guide.md`，含 7 张 Mermaid 流程图：架构总览、两层循环、Sub-Phase 执行、监控循环、过拟合处理、推进逻辑、State 数据结构。**Files**: `docs/Z1_Orchestrator_Guide.md` |
| 5 | Orchestrator 源码审阅 | 读取全部 7 个 automation 模块 (2357 行)，提取精确的内部流程写入文档 |
| 6 | 残留 orchestrator 进程确认 | 确认之前的 3 个 orchestrator 进程已退出，不占 GPU |
| 7 | 直接 torchrun vs orchestrator 分析 | 回答用户关于 embedded monitor 缺失的问题，给出 p3_coarse_v2 完成后接入 orchestrator 的方案 |

### Uncompleted / Blocked

| # | Item | Blocker | Next Step |
|---|------|---------|-----------|
| 1 | p3_coarse_v2 完成后接入 orchestrator | 训练仍在进行 (iter 13013/15000, ETA ~7.5h) | 等训练完成 → 更新 state → orchestrator --resume 从 p3_fine 开始 |
| 2 | 更新 bestmodel_phase.json | p3_coarse_v2 尚未完成 | 训练完成后更新 |

### Key Decisions

- Orchestrator 文档独立于 `Z1_5Phase_Auto_Pipeline.md`，后者专注参数表，新文档专注运维流程和故障恢复
- p3_coarse_v2 采用直接 torchrun 而非 orchestrator 启动（因之前 state 管理出问题），后续阶段重新接入 orchestrator
- 旧 run 目录加 `_old_v1` 后缀防止 stale TensorBoard 数据干扰
