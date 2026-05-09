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

---

## [18:30] Session Summary — Orchestrator `--adopt` 功能实现 + 命名统一

### Completed

| # | Item | Details |
|---|------|---------|
| 1 | `--adopt` CLI 参数 | `parse_args()` 新增 `--adopt` flag，`PhaseOrchestrator.__init__` 新增 `adopt` 参数 |
| 2 | `run()` adopt 分支 | 在 crash recovery 之后、`_init_new_run()` 之前插入 adopt 逻辑 |
| 3 | `_adopt_existing_training()` | 核心方法：检测运行中的训练 → 解析参数 → 推断子阶段 ID → 构建 state → 挂载 monitor |
| 4 | `_detect_running_training()` | `ps aux` + grep 找 train_multigpu/train.py 进程，支持 torchrun 父进程 → pgrep 找子进程 |
| 5 | `_parse_cmdline(pid)` | 读取 `/proc/PID/cmdline`，解析 `--key value` 参数对 |
| 6 | `_infer_sub_phase_id()` | 从 `agent_cfg` 路径推断（优先），或从 `run_name` 匹配 plan 中的子阶段 ID |
| 7 | 命名统一 `--automation` → `--orchestrator` | skill.md 全局替换，与代码文件 `phase_orchestrator.py` 保持一致 |
| 8 | skill.md 新增 `--orchestrator --adopt` 文档 | 子命令表 + 完整使用说明（pipeline、prerequisites、SSH 命令模板） |

### Modified Files

| File | Change |
|------|--------|
| `scripts/automation/phase_orchestrator.py` | +`--adopt` arg, +4 methods, +run() branch, +main() param |
| `~/.claude/skills/gpu-train/skill.md` | `--automation` → `--orchestrator` rename, +`--adopt` docs |

### Key Design Decisions

- **进程检测**：优先找 `train_multigpu.py` + `torchrun` 组合，pgrep 子进程获取实际训练参数
- **子阶段推断**：三级回退策略 — `agent_cfg` 路径 → `run_name` 精确匹配 → 去后缀匹配
- **adopt 后行为**：与正常 orchestrator run 完全一致 — 监控 → 过拟合检测 → 回滚 → 录视频 → 推进下一阶段
- **无需改 `_monitor_sub_phase`**：`self._proc is None` 时 `returncode=-1` 已能正确触发 `_check_max_iterations_reached()`

---

## [19:45] Session Summary — `--adopt` RTX 实测调试 + 成功接管 p3_coarse_v2

### Completed

| # | Item | Details |
|---|------|---------|
| 1 | Bug fix: `_parse_cmdline` 无法解析 `--key=value` | **Root cause**: torchrun 子进程的 cmdline 用 `--key=value` 格式（如 `--run_name=p3_coarse_v2`），而非 `--key value` · **Fix**: 加 `=` 分割逻辑，`split("=", 1)` 提取 key/value · **Files**: `phase_orchestrator.py` `_parse_cmdline()` |
| 2 | Bug fix: `_detect_running_training` 匹配到 bash wrapper | **Root cause**: bash -c wrapper 行包含 "train_multigpu.py" + "torchrun"，先于实际 python 进程被匹配 · **Fix**: 改为按优先级收集候选进程（worker > torchrun > single），跳过 bash wrapper · **Files**: `phase_orchestrator.py` `_detect_running_training()` |
| 3 | 第一次 adopt 失败后错误启动 p1_coarse 训练 | 发现后立即 kill 错误的 orchestrator (PID 255909) + 训练 (PID 255918)，原始 p3_coarse_v2 未受影响 |
| 4 | 修复后第二次 adopt 成功 | Orchestrator (PID 261671) 成功接管 torchrun (PID 48006)，state 文件 + monitor 均正常 |
| 5 | 验证 adopt 状态 | `orchestrator_state.json` 确认：sp=p3_coarse, pid=48006, status=running, monitor 首次 poll iter=10136 reward=36.05 HEALTHY |
| 6 | 上传修复后的代码到 RTX | `scp phase_orchestrator.py` + `python -c 'import ...'` 验证通过 |

### Uncompleted / Blocked

| # | Item | Blocker | Next Step |
|---|------|---------|-----------|
| 1 | p3_coarse_v2 训练完成 + orchestrator 自动推进 | 训练 iter ~10136/15000，ETA ~5h | Orchestrator 自动监控，完成后推进 p3_fine |
| 2 | 训练完成后的 post-processing（下载模型/视频） | 依赖训练完成 | `/gpu-train --orchestrator --status` 监控 |

### Key Decisions

- torchrun 的 cmdline 格式固定为 `--key=value`（`=` 分隔），而 `train.py` 用 `--key value`（空格分隔），两者都需支持
- 进程检测优先级：python worker（直接有 args）> torchrun launcher > single-GPU train.py > bash wrapper（跳过）
- adopt 失败时 fallthrough 到 `_init_new_run()` 是危险行为（会在有训练运行时启动新训练），应考虑加 guard — 当前通过 kill 快速处理了
