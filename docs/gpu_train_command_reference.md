# /gpu-train 命令速查表

> 所有命令通过 `/gpu-train --<command>` 在 Claude Code 中调用。
> 完整定义见 `~/.claude/skills/gpu-train/skill.md`。

---

## 监控与查看

| 命令 | 功能 | 备注 |
|------|------|------|
| `--status` | 检查训练进程是否存活 | SSH 检测 process |
| `--tail` | 最近训练日志（末尾 30 行） | 支持 Z1 / ARM101 子项目 |
| `--live` | 实时训练输出（tail -f） | |
| `--gpu` | GPU 使用情况 | `nvidia-smi` |
| `--idle` | 查找空闲 GPU | 建议可用 `--device cuda:X` |
| `--mycuda` | 仅显示自己的 CUDA 进程 | 过滤 user=phh |
| `--models` | 列出已保存的模型 checkpoint | |
| `--check` | 训练健康检查 | 趋势分析 + 可操作建议 |

## 训练监控（高级）

| 命令 | 功能 | 备注 |
|------|------|------|
| `--monitor` | 过拟合检测 + 最佳模型查找 | 解析 TensorBoard/reward 曲线，检测 5 种失败信号 |
| `--monitor_ARM101` | ACT Loss 监控（SO-ARM101） | ARM101 专用 |

## 视频与日志

| 命令 | 功能 | 备注 |
|------|------|------|
| `--sim` | 录制仿真视频（完整 pipeline） | Isaac Lab + MuJoCo sim2sim |
| `--compare` | 视频对比网格 | **已弃用**，用 `/merge` 替代 |
| `--training_log` | 记录本次会话工作总结 | 写入 `docs/training_logs/` |
| `--update` | 更新训练日志 | 追加到已有 log |

## 训练控制

| 命令 | 功能 | 备注 |
|------|------|------|
| `--start` | 启动/恢复训练 | 支持 GPU 交互选择 |
| `--kill` | 停止训练 | kill 训练进程 |
| `--local_play` | 本地 MuJoCo 键盘测试 | Z1 policy 查看器 |
| `--local_play_ARM101` | 本地 MuJoCo ACT policy 查看器 | SO-ARM101 专用 |

## 5-Phase Pipeline 自动化

| 命令 | 功能 | 备注 |
|------|------|------|
| `--orchestrator --start` | 启动 5-phase pipeline | 支持 `--from <SUB_PHASE>` |
| `--orchestrator --status` | 查看 pipeline 状态 + 当前进度 | |
| `--orchestrator --tail` | orchestrator 日志尾部 | |
| `--orchestrator --stop` | 停止 pipeline | kill orchestrator + 训练 |
| `--orchestrator --resume` | 从保存状态恢复 pipeline | |
| `--orchestrator --dry-run` | 干运行（打印 10 个 sub-phase 配置） | |
| `--orchestrator --adopt` | 接管当前运行中的训练 session | |
| `--rollback-update` | Pipeline rollback 后一键同步本地状态 | **新命令**，替代旧 `--sync-state` |

### `--rollback-update` 详情

Pipeline rollback 后自动同步所有本地文件：

1. SSH 读 `orchestrator_state.json` → 确定当前 sub-phase
2. 更新 `docs/tracking/bestmodel_phase.json`（归档旧 phase → PLANNED）
3. 归档旧 `videos/p/` → `_archived/`
4. 清理 `docs/github_readme/` 中已归档 phase 的图片
5. 更新 `README.md`（表格 + 移除已删除图片引用）

```bash
/gpu-train --rollback-update                    # 自动从 RTX 读取状态
/gpu-train --rollback-update --from p2_coarse   # 手动指定 rollback 点
```

## Slurm 集群

| 命令 | 功能 | 备注 |
|------|------|------|
| `--slurm` | Slurm 队列状态 + Bypass 检测 | 违规日志、GPU 冲突检测 |

## 连接

| 命令 | 功能 | 备注 |
|------|------|------|
| `--connect` | 测试 SSH 连接 | 失败自动启动 iNode VPN |

---

## 快速对照：常用场景

| 场景 | 命令 |
|------|------|
| 看看训练还在跑吗 | `/gpu-train --status` |
| 看最近日志 | `/gpu-train --tail` |
| 哪些 GPU 空闲 | `/gpu-train --idle` |
| 我占了哪些 GPU | `/gpu-train --mycuda` |
| 训练健康吗 | `/gpu-train --check` |
| 过拟合了吗 / 最佳模型 | `/gpu-train --monitor` |
| 启动训练 | `/gpu-train --start` |
| 停训练 | `/gpu-train --kill` |
| 启动自动化 pipeline | `/gpu-train --orchestrator --start` |
| Pipeline rollback 后同步 | `/gpu-train --rollback-update` |
| 录视频 | `/gpu-train --sim` |
| 本地看 policy | `/gpu-train --local_play` |
| 记录今天工作 | `/gpu-train --training_log` |
