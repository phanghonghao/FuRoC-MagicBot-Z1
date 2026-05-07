# Pipeline Smoke Test & Config Generator Bugfix

**Date**: 2026-05-07

---

## 1. `--smoke-test` 功能

**文件**: `scripts/automation/phase_orchestrator.py`

新增 CLI 参数 `--smoke-test`，走真实 pipeline 代码路径，但对每个 sub-phase 做最小化改动：

| 改动 | 正常模式 | Smoke 模式 |
|------|---------|-----------|
| `max_iterations` | 5000~10000 | **50** |
| `poll_interval` | 120s | **15s**（无需等待） |
| `num_envs` | 4096+ | **64**（避免 OOM） |

**执行流程**（对每个 sub-phase: p1_coarse → ... → p5_fine）：

1. Config 生成 — `generate_env_config()` + `generate_ppo_override()`
2. Swap config — 备份并替换 `velocity_env_cfg.py`
3. Launch training — `max_iterations=50, num_envs=64`
4. Wait run dir — 等待 Isaac Sim 创建 run directory（最多 5 分钟）
5. Wait checkpoint — 轮询直到至少 1 个 `model_*.pt` 出现（最多 3 分钟）
6. Stop training — `graceful_stop()`
7. 标记通过 → advance 到下一个 sub-phase

**结束行为**：

- 全部通过 → 打印 `SMOKE TEST PASSED`，清理所有 smoke run directories，退出码 0
- 任一阶段失败 → 打印失败原因，保留 run dir 用于诊断，退出码 1
- 不保存 `orchestrator_state.json` — smoke test 不影响正式状态
- Smoke run directory 以 `_smoke_` 后缀标识，结束后统一删除

**用法**：

```bash
# 在 RTX 上运行 smoke test
python -u scripts/automation/phase_orchestrator.py \
    --plan training_plans/z1_5phase_plan.yaml \
    --smoke-test --num-gpus 4

# 成功后正式启动 pipeline
python -u scripts/automation/phase_orchestrator.py \
    --plan training_plans/z1_5phase_plan.yaml \
    --fresh --num-gpus 4
```

**新增方法**：

| 方法 | 功能 |
|------|------|
| `_run_smoke_test()` | smoke test 主循环，遍历所有 sub-phase |
| `_smoke_run_sub_phase()` | 单个 sub-phase 的 smoke 验证 |
| `_cleanup_smoke_runs()` | 清理所有 smoke run directories |

**预计耗时**：每个 sub-phase ~100s（60s Isaac Sim 初始化 + 30s 跑 50 iter + 10s cleanup），10 个 sub-phase 共约 **17 分钟**。

---

## 2. config_generator.py 正则 Bug 修复

**文件**: `scripts/automation/config_generator.py`

### 根因

`_TEMPLATE_CFG_REL = _ACTIVE_CFG_REL`（第 28 行）— 模板和活动配置是同一个文件。

**Bug 链**：

1. `generate_env_config()` 从 `_ACTIVE_CFG_REL` 读取（活动配置路径）
2. 替换 #1 的正则 `r'^COBBLESTONE_ROAD_CFG = .*?^(\n)'` 需要空行标记 COBBLESTONE_ROAD_CFG 块的结束
3. 替换 #1 消耗了末尾的空行（作为匹配的一部分）
4. 下次生成时（如 p1_fine → p2_coarse），没有空行 → 正则一路匹配到 RobotSceneCfg 之后的空行 → 整个 `RobotSceneCfg` 类被吃掉
5. **p2-p5 全部受影响**，因为模板在每个 sub-phase 持续损坏

### 错误表现

```
NameError: name 'RobotSceneCfg' is not defined
  File "velocity_env_cfg.py", line 328, in RobotEnvCfg
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=16384, env_spacing=2.5)
```

### 修复

首次调用时存 `.orig` 备份，后续始终从原始模板读取：

```python
orig_path = template_path.parent / (template_path.name + ".orig")
if orig_path.exists():
    read_path = orig_path          # always read from pristine backup
else:
    shutil.copy2(str(template_path), str(orig_path))  # first call: snapshot
    read_path = template_path
```

### 验证结果

- 全部 **10 个 sub-phase** 生成均包含 `RobotSceneCfg` ✓
- 生成是**幂等的**（相同输入 → 相同 MD5 hash，不受中间运行影响）✓

---

## 3. 训练启动错误快速检测

**文件**: `scripts/automation/phase_orchestrator.py`

**之前**：训练进程因 NameError 秒崩，orchestrator 傻等 4 分钟超时才报 "Could not find run directory"。

**现在**：30 秒内检查训练日志是否出现 `Error`/`NameError`，立即标记失败，不再浪费等待时间。

---

## 4. 当前状态

p2_coarse 从 p1_fine 的 `model_2800.pt` 恢复启动，reward 41.61，HEALTHY，正在正常训练中。
