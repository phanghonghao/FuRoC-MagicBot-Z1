# Z1 5-Phase Orchestrator 运维指南

> Phase-Based Training Orchestrator 的完整技术文档，涵盖架构、流程、状态管理、CLI 用法和故障恢复。

## 1. 架构总览

```mermaid
graph TB
    subgraph "phase_orchestrator.py (主循环)"
        MAIN[PhaseOrchestrator.run]
    end

    subgraph "核心组件"
        PM[phase_manager.py<br/>YAML 解析 + 三层合并]
        CG[config_generator.py<br/>生成 velocity_env_cfg.py]
        PO[ppo_override.py<br/>生成 PPO 临时配置]
        TL[training_launcher.py<br/>torchrun 启动/停止]
        EM[embedded_monitor.py<br/>TensorBoard 轮询 + 过拟合检测]
        SS[state_store.py<br/>JSON 原子读写]
    end

    subgraph "外部依赖"
        TB[train_monitor.py<br/>OverfittingDetector 等]
        TM[train_multigpu.py<br/>多 GPU 训练入口]
    end

    MAIN --> PM
    MAIN --> CG
    MAIN --> PO
    MAIN --> TL
    MAIN --> EM
    MAIN --> SS
    EM --> TB
    TL -->|torchrun| TM

    style MAIN fill:#4a90d9,color:#fff
    style PM fill:#7b68ee,color:#fff
    style CG fill:#7b68ee,color:#fff
    style PO fill:#7b68ee,color:#fff
    style TL fill:#7b68ee,color:#fff
    style EM fill:#7b68ee,color:#fff
    style SS fill:#7b68ee,color:#fff
    style TB fill:#999,color:#fff
    style TM fill:#999,color:#fff
```

## 2. 两层循环：主事件循环

```mermaid
flowchart TD
    START([run]) --> RECOVER{crash recovery<br/>state 存在?}
    RECOVER -->|state == running<br/>PID 存活| RESUME[resume_monitor<br/>重新挂载 monitor]
    RECOVER -->|state == running<br/>PID 已死| FAIL_MARK[status = failed]
    RECOVER -->|state None / complete / failed| INIT[_init_new_run<br/>创建新 state]

    RESUME --> LOOP
    FAIL_MARK --> LOOP
    INIT --> LOOP

    subgraph LOOP["主事件循环 (while True)"]
        STATUS{current_stage_status?}

        STATUS -->|pending| START_SP[_start_sub_phase<br/>生成配置 → 启动训练]
        STATUS -->|running| MONITOR[_monitor_sub_phase<br/>轮询 monitor + 检查 PID]
        STATUS -->|overfitting| HANDLE[_handle_overfitting<br/>停止训练 → 回滚判断 → 录视频]
        STATUS -->|complete| ADVANCE[_advance<br/>推进到下一个 sub-phase]
        STATUS -->|failed| RETRY[_handle_failure<br/>重试或放弃]

        START_SP -->|status = running| SAVE[save state]
        MONITOR -->|monitor 触发 overfitting| SAVE
        HANDLE -->|status = complete| SAVE
        ADVANCE -->|status = pending| SAVE
        RETRY -->|retry < max → status = pending| SAVE
        RETRY -->|retry >= max → exit| EXIT2([sys.exit 1])

        SAVE --> SLEEP[sleep poll_interval]
        SLEEP --> STATUS
    end

    SAVE -->|KeyboardInterrupt| SAVE_EXIT[save state + exit]
    ADVANCE -->|无下一个 sub-phase| ALL_DONE([ALL COMPLETE<br/>sys.exit 0])

    style LOOP fill:#f0f7ff,stroke:#4a90d9
    style STATUS fill:#ffd700,stroke:#333
    style ALL_DONE fill:#2ecc71,color:#fff
    style EXIT2 fill:#e74c3c,color:#fff
```

## 3. Sub-Phase 执行流程 (pending → running)

```mermaid
flowchart TD
    PENDING["_start_sub_phase()"] --> GEN_ENV["1. config_generator<br/>生成 velocity_env_cfg.py<br/>→ tmp/phase_configs/{sp_id}/"]
    GEN_ENV --> SWAP["2. 备份 + 替换<br/>velocity_env_cfg.py<br/>(.bak.{sp_id} 备份)"]
    SWAP --> GEN_PPO["3. ppo_override<br/>生成 ppo_override_cfg.py<br/>(LR, entropy 覆盖)"]
    GEN_PPO --> RESOLVE["4. _resolve_checkpoint<br/>从 stage_history 找上一个 best"]
    RESOLVE --> LAUNCH["5. training_launcher.launch<br/>torchrun --nproc_per_node=N<br/>--agent_cfg=ppo_override_cfg.py"]

    LAUNCH --> WAIT_DIR["6. 等待 run 目录创建<br/>(最多 8 × 30s = 4min)"]
    WAIT_DIR -->|找到| SETUP_MON["7. 初始化 EmbeddedMonitor<br/>设置 terrain / threshold"]
    WAIT_DIR -->|超时| FAIL["status = failed"]

    SETUP_MON --> UPDATE_STATE["8. 更新 state<br/>PID + run_dir + status=running"]
    UPDATE_STATE --> RUNNING([status: running])

    style PENDING fill:#4a90d9,color:#fff
    style RUNNING fill:#2ecc71,color:#fff
    style FAIL fill:#e74c3c,color:#fff
```

## 4. 监控循环 (running → overfitting)

```mermaid
flowchart TD
    MON["_monitor_sub_phase()"] --> PID_ALIVE{PID 存活?}

    PID_ALIVE -->|No, returncode=0| EXIT_NORM["训练正常退出"]
    PID_ALIVE -->|No, returncode≠0| MAX_ITER{达到 max_iterations?}
    PID_ALIVE -->|Yes| POLL["monitor.poll()"]

    MAX_ITER -->|Yes| EXIT_NORM
    MAX_ITER -->|No| CRASH["status = failed"]

    EXIT_NORM --> FINAL_POLL["最终 poll"]
    FINAL_POLL --> OVERFIT["status = overfitting"]

    POLL --> SUMMARY{"poll 结果"}
    SUMMARY -->|"NO_DATA"| WAIT[sleep poll_interval]
    SUMMARY -->|"HEALTHY"| LOG["日志: iter, reward, peak, best"]
    SUMMARY -->|"OVERFITTING"| CALLBACK["_on_overfitting_callback<br/>status = overfitting"]

    WAIT --> MON
    LOG --> WAIT2[sleep poll_interval]
    WAIT2 --> MON

    style OVERFIT fill:#f39c12,color:#fff
    style CRASH fill:#e74c3c,color:#fff
    style CALLBACK fill:#f39c12,color:#fff
```

## 5. 过拟合处理 + 回滚 (overfitting → complete)

```mermaid
flowchart TD
    HANDLE["_handle_overfitting()"] --> STOP["1. 停止训练 PID"]
    STOP --> FINAL["2. 最终 monitor.poll()<br/>获取 best checkpoint"]

    FINAL --> ROLLBACK_CHECK{"3. 回滚判断<br/>best_reward < starting × 0.95?"}

    ROLLBACK_CHECK -->|"Yes + retry < max"| REDUCE_LR["降低 LR × 0.5<br/>重写 ppo_override_cfg.py"]
    REDUCE_LR --> RETRY["status = pending<br/>retry_count += 1"]

    ROLLBACK_CHECK -->|"No / retry 耗尽"| RECORD["4. 记录到 stage_history"]
    RECORD --> POST["5. _run_post_phase_pipeline"]
    POST --> PHASE_CHECK["6. _check_phase_completion<br/>是否 phase 最后一个 sub-phase?"]
    PHASE_CHECK --> COMPLETE["status = complete"]

    subgraph "Post-Phase Pipeline"
        POST --> JIT["JIT Export<br/>export_jit.py"]
        JIT --> MUJOCO["MuJoCo 录像<br/>mujoco_manual.py"]
        POST --> ISAAC["Isaac Sim 录像<br/>play.py --video"]
        POST --> PLOTS["生成 plots<br/>plot_learning_curves.py"]
    end

    style RETRY fill:#f39c12,color:#fff
    style COMPLETE fill:#2ecc71,color:#fff
    style POST fill:#e8f4fd,stroke:#4a90d9
```

## 6. 推进逻辑 (complete → next pending)

```mermaid
flowchart LR
    ADV["_advance()"] --> NEXT{"get_next_sub_phase()"}

    NEXT -->|"p1_coarse → p1_fine"| P1F["p1_fine<br/>(同 phase)"]
    NEXT -->|"p1_fine → p2_coarse"| P2C["p2_coarse<br/>(新 phase)"]
    NEXT -->|"p5_fine → None"| DONE([ALL COMPLETE])

    P1F --> SET["status = pending<br/>PID = None<br/>rollback = 0"]
    P2C --> SET
    SET --> LOOP([回到主循环])

    style DONE fill:#2ecc71,color:#fff
    style LOOP fill:#4a90d9,color:#fff
```

## 7. 状态管理 (orchestrator_state.json)

### 7.1 State 数据结构

```mermaid
classDiagram
    class OrchestratorState {
        +String plan_name
        +String current_stage_id        "当前 sub-phase (如 p3_coarse)"
        +String current_stage_status    "pending | running | overfitting | complete | failed"
        +int training_pid               "训练进程 PID"
        +String training_run_dir        "训练 run 目录路径"
        +String best_checkpoint_path    "当前 best checkpoint 路径"
        +float best_reward              "当前 best reward"
        +List stage_history             "已完成 sub-phase 结果列表"
        +int retry_count                "失败重试计数"
        +String started_at              "启动时间 ISO"
        +String updated_at              "最后更新时间 ISO"
        +String current_phase_id        "当前 phase (如 p3)"
        +float starting_reward          "sub-phase 开始时 reward (用于回滚判断)"
        +int rollback_count             "回滚计数"
        +String error_message           "最后错误信息"
        +List phase_history             "已完成 phase 结果列表"
    }

    class StageHistoryEntry {
        +String sub_phase_id
        +String phase_id
        +String status
        +String best_checkpoint_path
        +float best_reward
        +float starting_reward
        +int rollback_count
        +String training_run_dir
        +String completed_at
    }

    OrchestratorState --> StageHistoryEntry : stage_history
```

### 7.2 State 写入时机

| 时机 | 写入内容 |
|------|----------|
| `_init_new_run` | plan_name, start_id, status=pending |
| `_start_sub_phase` | PID, run_dir, status=running, phase_id |
| `_handle_overfitting` | stage_history 追加, best_checkpoint, best_reward |
| `_advance` | current_stage_id=next, status=pending |
| `_handle_failure` | retry_count++ |
| `KeyboardInterrupt` | 保存当前 state |
| 主循环每轮结束 | save (原子写入) |

### 7.3 原子写入

StateStore 使用 write-to-tmp + os.replace 模式，确保断电不会损坏文件。

## 8. CLI 参数与模式

### 8.1 启动模式决策树

```mermaid
flowchart TD
    CLI["CLI 调用"] --> FRESH{--fresh?}
    FRESH -->|Yes| CLEAR["清除 state 文件<br/>start_id = --start-from 或 p1_coarse"]
    FRESH -->|No| SF{--start-from?}
    SF -->|Yes| START_FROM["start_id = --start-from"]
    SF -->|No| DEFAULT["start_id = p1_coarse"]

    CLEAR --> RUN[run]
    START_FROM --> RUN
    DEFAULT --> RUN

    RUN --> LOAD{加载 state}
    LOAD -->|state 存在 + running<br/>+ PID 存活| RECOVER["crash recovery<br/>重新挂载 monitor"]
    LOAD -->|state 存在 + running<br/>+ PID 已死| DEAD["status = failed"]
    LOAD -->|state None / complete / failed| INIT["_init_new_run<br/>用 start_id 创建新 state"]

    RECOVER --> LOOP([进入主循环])
    DEAD --> LOOP
    INIT --> LOOP

    style CLI fill:#4a90d9,color:#fff
    style RECOVER fill:#f39c12,color:#fff
    style INIT fill:#2ecc71,color:#fff
```

### 8.2 模式对比

| 模式 | CLI | 行为 | 使用场景 |
|------|-----|------|----------|
| **Fresh** | `--fresh` | 删除 state，从 p1_coarse 开始 | 全新训练、彻底重开 |
| **Fresh + Start From** | `--fresh --start-from p3_coarse` | 删除 state，从指定 sub-phase 开始（无前序 checkpoint 信息） | 跳过已完成阶段，但需要手动提供 checkpoint |
| **Start From** | `--start-from p3_coarse` | 保留 state，从指定 sub-phase 开始 | 跳到特定阶段 |
| **Resume** | (无特殊参数) | 加载 state，继续当前进度 | orchestrator 崩溃后恢复 |
| **Dry Run** | `--dry-run` | 打印所有 10 个 sub-phase 参数，不执行 | 参数检查 |
| **Smoke Test** | `--smoke-test` | 每个 sub-phase 只跑 50 iter 验证管线 | 管线完整性验证 |

> **注意**: `--fresh --start-from` 会导致 `_resolve_checkpoint()` 返回 None（因为 state 被清空、stage_history 为空），训练将从随机初始化开始。必须手动指定 `--load_run` 和 `--checkpoint`，或先让前序 phase 完成以积累 stage_history。

### 8.3 完整 CLI 参数

```
--plan           训练计划 YAML 路径 (required)
--project-root   项目根目录 (default: .)
--log-root       TensorBoard 日志根目录 (default: logs/rsl_rl/...)
--start-from     起始 sub-phase ID
--fresh          忽略已保存状态
--dry-run        打印配置不执行
--smoke-test     每个 sub-phase 最小化验证
--poll-interval  监控轮询间隔秒数 (default: 120)
--state-file     state 文件名 (default: orchestrator_state.json)
--num-gpus       GPU 数量 (default: 4)
```

## 9. EmbeddedMonitor 内部流程

```mermaid
flowchart TD
    POLL["monitor.poll()"] --> READ_TB["1. 增量读取 TensorBoard<br/>(last_checked_step 之后)"]
    READ_TB --> READ_CKPT["2. 扫描新 checkpoint 文件<br/>提取 policy std"]
    READ_CKPT --> UPDATE["3. 更新 peak_reward<br/>best_model tracking"]
    UPDATE --> DETECT["4. OverfittingDetector.check()"]

    DETECT --> SIGNALS{"5 个独立信号<br/>(任一触发即告警)"}
    SIGNALS --> S1["reward decline > 20%"]
    SIGNALS --> S2["action_rate < threshold"]
    SIGNALS --> S3["policy std < 0.01"]
    SIGNALS --> S4["value_loss > 100"]
    SIGNALS --> S5["entropy 下降 > 80%"]

    S1 & S2 & S3 & S4 & S5 --> RESULT{"检测结果"}
    RESULT -->|有信号 + 首次| FIRE["触发 on_overfitting callback<br/>→ orchestrator status = overfitting"]
    RESULT -->|无信号| HEALTHY["返回 HEALTHY"]

    style FIRE fill:#e74c3c,color:#fff
    style HEALTHY fill:#2ecc71,color:#fff
```

## 10. 自动回滚机制

```mermaid
flowchart TD
    OVERFIT["overfitting 检测"] --> GET_BEST["获取 best_reward"]
    GET_BEST --> CHECK{"best_reward < starting_reward × 0.95?"}

    CHECK -->|Yes| COUNT{"rollback_count < max_retries (1)?"}
    COUNT -->|Yes| ROLLBACK["回滚流程:<br/>1. LR × 0.5<br/>2. 重写 ppo_override_cfg.py<br/>3. status = pending<br/>4. rollback_count += 1"]
    COUNT -->|No| EXHAUSTED["回滚次数耗尽<br/>记录为 rollback_exhausted<br/>仍推进到下一阶段"]

    CHECK -->|No| NORMAL["正常推进"]

    ROLLBACK --> RELAUNCH([回到 pending → 重新启动训练])
    EXHAUSTED --> RECORD["记录到 stage_history"]
    NORMAL --> RECORD

    style ROLLBACK fill:#f39c12,color:#fff
    style NORMAL fill:#2ecc71,color:#fff
    style EXHAUSTED fill:#e74c3c,color:#fff
```

## 11. 故障恢复与运维

### 11.1 常见故障场景

```mermaid
flowchart TD
    subgraph "故障场景"
        F1["Orchestrator 进程崩溃<br/>(OOM / 断电 / Ctrl+C)"]
        F2["训练进程崩溃<br/>(CUDA error / 超时)"]
        F3["Stale TensorBoard 数据<br/>误判过拟合"]
        F4["State 文件过期<br/>指向旧 run"]
        F5["--fresh --start-from<br/>无 checkpoint 信息"]
    end

    subgraph "恢复方案"
        R1["--resume (无参数)<br/>自动从 state 恢复"]
        R2["手动 torchrun 启动<br/>直接指定 --load_run --checkpoint"]
        R3["重命名旧 run 目录<br/>添加 _old_vN 后缀"]
        R4["手动编辑 state JSON<br/>或删除后用 --start-from"]
        R5["直接 torchrun 指定<br/>--load_run + --checkpoint"]
    end

    F1 --> R1
    F2 --> R1
    F3 --> R3
    F4 --> R4
    F5 --> R5

    style F3 fill:#e74c3c,color:#fff
    style F4 fill:#e74c3c,color:#fff
    style F5 fill:#e74c3c,color:#fff
```

### 11.2 恢复操作手册

#### 场景 A: Orchestrator 崩溃，训练仍在运行

```bash
# 1. 确认训练进程存活
ssh phh@192.168.120.155 'ps aux | grep train_multigpu | grep phh | grep -v grep'

# 2. 直接 resume orchestrator（会自动挂载到运行中的训练）
ssh phh@192.168.120.155 "cd ~/magiclab_rl_lab && nohup python -u scripts/automation/phase_orchestrator.py \
    --plan training_plans/z1_5phase_plan.yaml --num-gpus 4 --poll-interval 120 \
    > /tmp/z1_5phase_pipeline.log 2>&1 & echo PID=\$!"
```

#### 场景 B: Orchestrator + 训练都崩溃

```bash
# 1. 检查 state 文件确认进度
ssh phh@192.168.120.155 'cat ~/magiclab_rl_lab/orchestrator_state.json'

# 2. 找到最新 checkpoint
ssh phh@192.168.120.155 'ls -t ~/magiclab_rl_lab/logs/rsl_rl/magiclab_z1_12dof_velocity/*p3_coarse*/model_*.pt | head -1'

# 3. Resume orchestrator（从 state 记录的位置继续）
ssh phh@192.168.120.155 "cd ~/magiclab_rl_lab && nohup python -u scripts/automation/phase_orchestrator.py \
    --plan training_plans/z1_5phase_plan.yaml --num-gpus 4 --poll-interval 120 \
    > /tmp/z1_5phase_pipeline.log 2>&1 & echo PID=\$!"
```

#### 场景 C: Stale TensorBoard 数据导致误判

```bash
# 1. 重命名旧的 run 目录（加 _old_vN 后缀）
ssh phh@192.168.120.155 'cd ~/magiclab_rl_lab/logs/rsl_rl/magiclab_z1_12dof_velocity && \
    for d in *p3_coarse* *p3_fine* *p4* *p5*; do
        [ -d "$d" ] && [ ! "${d##*_old_v*}" ] || mv "$d" "${d}_old_v1"
    done'

# 2. 重新启动 orchestrator
```

#### 场景 D: 绕过 Orchestrator，直接 torchrun

当 orchestrator state 管理出问题，或需要使用自定义参数时：

```bash
ssh phh@192.168.120.155 "cd ~/magiclab_rl_lab && source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab && \
    nohup torchrun --nproc_per_node=4 --master_port=29502 \
    scripts/rsl_rl/train_multigpu.py \
    --task=Magiclab-Z1-12dof-Velocity \
    --run_name=p3_coarse_v2 \
    --headless --distributed \
    --num_envs=4096 --max_iterations=15000 \
    --resume \
    --load_run=2026-05-07_03-10-20_p2_fine \
    --checkpoint=model_3000.pt \
    --agent_cfg=/home/phh/magiclab_rl_lab/tmp/phase_configs/p3_coarse/ppo_override_cfg.py \
    > /tmp/z1_p3_coarse_v2.log 2>&1 & echo PID=\$!"
```

> **注意**: 直接 torchrun 启动时**没有 embedded monitor**，需要手动使用 `--monitor` 监控或等训练完成后手动启动 orchestrator。

#### 场景 E: 手动修复 State 后 Resume

```bash
# 1. 备份当前 state
ssh phh@192.168.120.155 'cp ~/magiclab_rl_lab/orchestrator_state.json ~/magiclab_rl_lab/orchestrator_state.json.bak'

# 2. 编辑 state：更新 current_stage_id, best_checkpoint_path, best_reward 等
# 3. 启动 orchestrator（不带 --fresh，让它读修改后的 state）
```

### 11.3 p3_coarse_v2 完成后接入 Orchestrator

当前 p3_coarse_v2 是直接 torchrun 启动的，完成后需要手动更新 state 再启动 orchestrator 处理后续阶段：

```bash
# 1. 从 TensorBoard 找到 v2 的 best checkpoint
# 2. 手动更新 orchestrator_state.json：
#    - current_stage_id = "p3_fine"
#    - current_stage_status = "pending"
#    - best_checkpoint_path = "<v2 run dir>/model_<best>.pt"
#    - best_reward = <v2 best reward>
#    - starting_reward = <v2 best reward>
#    - stage_history 追加 p3_coarse v2 的结果
# 3. 启动 orchestrator（不带 --fresh）：
ssh phh@192.168.120.155 "cd ~/magiclab_rl_lab && nohup python -u scripts/automation/phase_orchestrator.py \
    --plan training_plans/z1_5phase_plan.yaml --num-gpus 4 --poll-interval 120 \
    > /tmp/z1_5phase_pipeline.log 2>&1 & echo PID=\$!"
```

## 12. 文件路径速查

| 文件 | 路径 (相对于 ~/magiclab_rl_lab/) |
|------|------|
| Orchestrator 主脚本 | `scripts/automation/phase_orchestrator.py` |
| Phase Manager | `scripts/automation/phase_manager.py` |
| Config Generator | `scripts/automation/config_generator.py` |
| PPO Override | `scripts/automation/ppo_override.py` |
| State Store | `scripts/automation/state_store.py` |
| Training Launcher | `scripts/automation/training_launcher.py` |
| Embedded Monitor | `scripts/automation/embedded_monitor.py` |
| Train Monitor | `scripts/train_monitor.py` |
| State 文件 | `orchestrator_state.json` |
| Pipeline 日志 | `/tmp/z1_5phase_pipeline.log` |
| Orchestrator 持久日志 | `logs/phase_orchestrator.log` |
| 生成的 env 配置 | `tmp/phase_configs/{sp_id}/velocity_env_cfg.py` |
| 生成的 PPO 配置 | `tmp/phase_configs/{sp_id}/ppo_override_cfg.py` |
| 训练 run 目录 | `logs/rsl_rl/magiclab_z1_12dof_velocity/<run_dir>/` |
| 视频 | `videos/phase_pipeline/{sp_id}.mp4` |
| Plots | `plots/{sp_id}/` |

## 13. 完整生命周期

```mermaid
flowchart LR
    subgraph "Phase 1"
        P1C[p1_coarse] --> P1F[p1_fine]
    end
    subgraph "Phase 2"
        P2C[p2_coarse] --> P2F[p2_fine]
    end
    subgraph "Phase 3"
        P3C[p3_coarse] --> P3F[p3_fine]
    end
    subgraph "Phase 4"
        P4C[p4_coarse] --> P4F[p4_fine]
    end
    subgraph "Phase 5"
        P5C[p5_coarse] --> P5F[p5_fine]
    end

    P1F --> P2C
    P2F --> P3C
    P3F --> P4C
    P4F --> P5C
    P5F --> DEPLOY([最终部署模型])

    style P1C fill:#3498db,color:#fff
    style P1F fill:#2980b9,color:#fff
    style P2C fill:#2ecc71,color:#fff
    style P2F fill:#27ae60,color:#fff
    style P3C fill:#f39c12,color:#fff
    style P3F fill:#e67e22,color:#fff
    style P4C fill:#e74c3c,color:#fff
    style P4F fill:#c0392b,color:#fff
    style P5C fill:#9b59b6,color:#fff
    style P5F fill:#8e44ad,color:#fff
    style DEPLOY fill:#2ecc71,color:#fff
```

每个 sub-phase 内部的完整流程:

```
config_gen → swap_env → ppo_gen → resolve_ckpt → torchrun → monitor_poll
→ overfitting → stop → rollback_check → record_video → advance
```

共 10 个 sub-phase，~210K iterations，预计总训练时间 28-35 小时 (4 GPU)。
