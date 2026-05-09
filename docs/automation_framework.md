# Automation Framework Architecture

> `scripts/automation/` — 5-Phase Automated RL Training Pipeline

## 1. Module Overview

```mermaid
graph TB
    subgraph "Entry Point"
        MAIN["__main__.py<br/>python -m automation"]
        CLI["CLI Arguments<br/>--plan --fresh --dry-run<br/>--start-from --smoke-test --adopt"]
    end

    subgraph "Orchestrator Core"
        PO["phase_orchestrator.py<br/>PhaseOrchestrator<br/>两层级事件循环"]
    end

    subgraph "Plan & Config"
        PM["phase_manager.py<br/>PhaseManager<br/>YAML 解析 + 三层合并"]
        CG["config_generator.py<br/>正则替换生成<br/>velocity_env_cfg.py"]
        PP["ppo_override.py<br/>生成 PPO 覆盖配置"]
    end

    subgraph "Training Launch"
        TL["training_launcher.py<br/>TrainingLauncher<br/>单GPU / 多GPU(torchrun)"]
        TRAIN["train.py / train_multigpu.py<br/>Isaac Lab 训练脚本"]
    end

    subgraph "Monitoring"
        EM["embedded_monitor.py<br/>EmbeddedMonitor<br/>轮询式监控接口"]
        TM["train_monitor.py<br/>OverfittingDetector<br/>5 信号过拟合检测"]
        TB["TensorBoard Events"]
        CKPT["Checkpoints<br/>model_N.pt"]
    end

    subgraph "State Persistence"
        SS["state_store.py<br/>StateStore<br/>原子 JSON 读写"]
        STATE["orchestrator_state.json"]
    end

    YAML["training_plans/<br/>z1_5phase_plan.yaml"]

    CLI --> MAIN
    MAIN --> PO
    YAML --> PM
    PM --> PO
    PO --> CG
    PO --> PP
    PO --> EM
    PO --> TL
    PO --> SS

    CG -->|生成| ENVCFG["velocity_env_cfg.py"]
    PP -->|生成| PPOCFG["ppo_override_cfg.py"]

    TL -->|启动| TRAIN
    EM -->|复用| TM
    TB -->|读取| EM
    CKPT -->|扫描| EM

    SS -->|持久化| STATE
    STATE -->|崩溃恢复| PO

    style PO fill:#4A90D9,color:#fff,stroke:#2C5F9E
    style PM fill:#7B68EE,color:#fff,stroke:#5B48CE
    style EM fill:#E67E22,color:#fff,stroke:#C0601A
    style TL fill:#27AE60,color:#fff,stroke:#1E8C4D
    style SS fill:#8E44AD,color:#fff,stroke:#6E349D
    style YAML fill:#F39C12,color:#fff,stroke:#D3800E
    style TRAIN fill:#2ECC71,color:#fff,stroke:#1EBC61
```

## 2. Sub-Phase Lifecycle (State Machine)

每个子阶段 (sub-phase) 的完整生命周期：

```mermaid
stateDiagram-v2
    [*] --> pending : start_from YAML / resume state

    pending --> running : _start_sub_phase()
    note right of pending
        1. PhaseManager 获取 SubPhaseConfig
        2. config_generator 生成 env cfg
        3. ppo_override 生成 PPO cfg
        4. _resolve_checkpoint() 找起始模型
        5. TrainingLauncher.launch() 启动训练
        6. EmbeddedMonitor.start() + reset_for_new_phase()
    end note

    running --> running : _monitor_sub_phase() poll every 120s
    note right of running
        EmbeddedMonitor.poll():
        - 增量读取 TensorBoard
        - 扫描新 Checkpoints
        - 更新 peak/best tracking
        - OverfittingDetector.check(phase_start_iter)
    end note

    running --> overfitting : 检测到过拟合 / 进程正常退出
    running --> failed : 进程异常退出 (非 max_iterations)

    overfitting --> rollback_check : _handle_overfitting()
    note right of overfitting
        1. graceful_stop(pid)
        2. 最终 poll 获取 best checkpoint
        3. 记录 stage_history
    end note

    rollback_check --> pending : best_reward < start × 0.95<br/>LR × 0.5 重试 (max 1次)
    rollback_check --> complete : 通过检查

    complete --> next_pending : _advance()
    complete --> [*] : 所有子阶段完成

    failed --> pending : retry_count < 2
    failed --> [*] : retry 耗尽 → sys.exit(1)

    next_pending --> running : 下一个子阶段
```

## 3. Main Event Loop

Orchestrator 主循环的完整流程：

```mermaid
flowchart TD
    START(["[启动]"]) --> LOAD_STATE{"加载 state.json"}
    LOAD_STATE -->|"有state, running"| RECOVER{"PID存活?"}
    LOAD_STATE -->|"有state, pending"| LOOP
    LOAD_STATE -->|"有state, complete"| FRESH["初始化新run"]
    LOAD_STATE -->|"无state"| FRESH

    RECOVER -->|"是"| RESUME_MON["_resume_monitor"]
    RECOVER -->|"否"| MARK_FAIL["status = failed"]
    RESUME_MON --> LOOP
    MARK_FAIL --> LOOP

    FRESH --> LOOP

    LOOP{{"主循环 poll=120s"}}
    LOOP --> STATUS{"检查 status"}

    STATUS -->|"pending"| LAUNCH["_start_sub_phase"]
    STATUS -->|"running"| MONITOR["_monitor_sub_phase"]
    STATUS -->|"overfitting"| HANDLE_OF["_handle_overfitting"]
    STATUS -->|"complete"| ADVANCE["_advance"]
    STATUS -->|"failed"| HANDLE_FAIL["_handle_failure"]

    LAUNCH --> GEN_ENV["config_generator 生成env cfg"]
    GEN_ENV --> GEN_PPO["ppo_override 生成PPO cfg"]
    GEN_PPO --> RESOLVE_CKPT["_resolve_checkpoint 找起始模型"]
    RESOLVE_CKPT --> LAUNCH_TRAIN["TrainingLauncher.launch torchrun"]
    LAUNCH_TRAIN --> WAIT_DIR["等待IsaacSim创建run dir"]
    WAIT_DIR --> SETUP_MON["Monitor.start + reset"]
    SETUP_MON -->|"status=running"| LOOP

    MONITOR --> POLL["EmbeddedMonitor.poll"]
    POLL --> CHECK_PROC{"进程存活?"}
    CHECK_PROC -->|"否,正常退出"| MARK_OF["status = overfitting"]
    CHECK_PROC -->|"否,到达上限"| MARK_OF
    CHECK_PROC -->|"否,异常"| MARK_FAIL2["status = failed"]
    CHECK_PROC -->|"是"| CHECK_OF{"过拟合?"}
    CHECK_OF -->|"是"| MARK_OF
    CHECK_OF -->|"否,HEALTHY"| LOG["日志输出 iter/reward/peak"]
    MARK_OF --> LOOP
    MARK_FAIL2 --> LOOP
    LOG --> LOOP

    HANDLE_OF --> STOP["graceful_stop(pid)"]
    STOP --> FINAL_POLL["最终poll获取best ckpt"]
    FINAL_POLL --> ROLLBACK{"reward退化超5%?"}
    ROLLBACK -->|"是,retry未耗尽"| RETRY["LR减半重试"]
    ROLLBACK -->|"否,通过"| SAVE_HIST["记录stage_history"]
    RETRY --> LOOP
    SAVE_HIST --> POST_PHASE["post_phase: JIT/MuJoCo/Isaac video"]
    POST_PHASE --> PHASE_CHECK{"phase最后一个?"}
    PHASE_CHECK --> MARK_COMPLETE["status = complete"]
    MARK_COMPLETE --> LOOP

    ADVANCE --> NEXT{"还有下一个子阶段?"}
    NEXT -->|"是"| SET_NEXT["设置next sub_phase"]
    NEXT -->|"否"| ALL_DONE(["ALL COMPLETE"])
    SET_NEXT --> LOOP

    HANDLE_FAIL --> RETRY_FAIL{"retry 小于 2?"}
    RETRY_FAIL -->|"是"| RETRY_SET["retry++, status=pending"]
    RETRY_FAIL -->|"否"| EXIT_FAIL(["sys.exit 1"])
    RETRY_SET --> LOOP

    style LOOP fill:#4A90D9,color:#fff,stroke:#2C5F9E
    style LAUNCH fill:#27AE60,color:#fff
    style POLL fill:#E67E22,color:#fff
    style HANDLE_OF fill:#E74C3C,color:#fff
    style POST_PHASE fill:#9B59B6,color:#fff
```

## 4. Config Generation Pipeline

从 YAML 到可训练配置的生成流程：

```mermaid
flowchart LR
    subgraph "YAML Plan"
        YAML["z1_5phase_plan.yaml"]
        DEFAULTS["defaults:<br/>env / ppo / monitor"]
        PHASE["phase p3b:<br/>env / ppo / rewards"]
        SUBPHASE["sub_phase p3b_coarse:<br/>env / ppo / rewards / monitor"]
    end

    YAML --> PM

    subgraph "PhaseManager — 三层合并"
        PM["PhaseManager"]
        MERGE["_deep_merge × 2<br/>defaults → phase → sub_phase"]
    end

    PM --> MERGE
    MERGE --> MERGED["SubPhaseConfig<br/>(env, ppo, rewards, monitor<br/>全部展开合并)"]

    MERGED --> CG
    MERGED --> PP

    subgraph "config_generator.py"
        CG["generate_env_config()"]
        CG1["1. 备份模板 → .orig"]
        CG2["2. 正则替换 8 个块:<br/>terrain_generator<br/>terrain_scene<br/>commands<br/>rewards<br/>terminations<br/>action_scale<br/>sim_params<br/>curriculum"]
        CG --> CG1 --> CG2
    end

    subgraph "ppo_override.py"
        PP["generate_ppo_override()"]
        PP1["生成 PhasePPORunnerCfg:<br/>policy (actor/critic dims)<br/>algorithm (LR, entropy, clip)<br/>training (steps, epochs, batches)"]
        PP --> PP1
    end

    CG2 --> ENV_OUT["tmp/phase_configs/<br/>p3b_coarse/velocity_env_cfg.py"]
    PP1 --> PPO_OUT["tmp/phase_configs/<br/>p3b_coarse/ppo_override_cfg.py"]

    ENV_OUT --> SWAP["swap active config<br/>(覆盖 source/.../velocity_env_cfg.py)"]
    PPO_OUT --> AGENT["--agent_cfg 传入训练脚本"]

    style MERGE fill:#7B68EE,color:#fff
    style CG fill:#27AE60,color:#fff
    style PP fill:#27AE60,color:#fff
```

## 5. Overfitting Detection Pipeline

EmbeddedMonitor 的增量检测流程：

```mermaid
flowchart TD
    POLL["EmbeddedMonitor.poll()"] --> READ_TB["增量读取 TensorBoard<br/>(last_checked_step 之后)"]
    READ_TB --> EXTEND["扩展 rewards / action_rates<br/>value_losses / entropies"]
    EXTEND --> SCAN_CKPT["扫描新 Checkpoints<br/>提取 std values"]
    SCAN_CKPT --> UPDATE_PEAK["更新 peak_reward<br/>peak_entropy<br/>BestModelTracker"]

    UPDATE_PEAK --> BASELINE{首次数据?}
    BASELINE -->|是| SET_BASE["设置 phase_start_iter<br/>设置 baseline_iter"]
    BASELINE -->|否| DETECT

    SET_BASE --> DETECT

    DETECT["OverfittingDetector.check<br/>(phase_start_iter)"]

    DETECT --> GATE{"relative_iter<br/> < min_iterations?"}
    GATE -->|是| HEALTHY["return None (跳过检测)"]
    GATE -->|否| CHECK1

    CHECK1["1. Reward Decline<br/>median(last 10) < peak × (1-pct)?"]
    CHECK1 -->|触发| ALERT
    CHECK1 -->|正常| CHECK2["2. Action Rate<br/>median(last 10) < threshold?"]
    CHECK2 -->|触发| ALERT
    CHECK2 -->|正常| CHECK3["3. Std Collapse<br/>latest_std < 0.01?"]
    CHECK3 -->|触发| ALERT
    CHECK3 -->|正常| CHECK4["4. Value Loss<br/>median(last 5) > 100?"]
    CHECK4 -->|触发| ALERT
    CHECK4 -->|正常| CHECK5["5. Entropy Collapse<br/>drop > 95% from peak?"]
    CHECK5 -->|触发| ALERT
    CHECK5 -->|正常| HEALTHY

    ALERT["返回 reason string<br/>(任意一个触发即告警)"]
    ALERT --> FIRED{已触发过?}
    FIRED -->|否| CALLBACK["on_overfitting callback<br/>→ orchestrator 标记 status=overfitting"]
    FIRED -->|是| NOOP["不再重复触发"]

    HEALTHY --> RETURN["返回 HEALTHY summary"]
    CALLBACK --> RETURN_OF["返回 OVERFITTING summary"]

    style DETECT fill:#E67E22,color:#fff
    style ALERT fill:#E74C3C,color:#fff
    style HEALTHY fill:#27AE60,color:#fff
```

## 6. Rollback & Phase Transition

过拟合处理后的回滚判断与阶段推进：

```mermaid
flowchart TD
    OF["过拟合检测触发"] --> STOP["graceful_stop(pid)<br/>SIGTERM → 5s → SIGKILL"]
    STOP --> FINAL["最终 EmbeddedMonitor.poll()<br/>获取 best checkpoint"]

    FINAL --> BEST_CKPT{"找到<br/>best checkpoint?"}
    BEST_CKPT -->|是| USE_BEST["使用 best checkpoint"]
    BEST_CKPT -->|否| FALLBACK["fallback: 最新 checkpoint"]

    USE_BEST --> ROLLBACK_CHECK
    FALLBACK --> ROLLBACK_CHECK

    ROLLBACK_CHECK{"best_reward <br/>< starting_reward × 0.95?"}

    ROLLBACK_CHECK -->|"是 (退化)"| RETRY_CHECK{"rollback_count<br/>< max_retries (1)?"}
    RETRY_CHECK -->|是| REDUCE_LR["LR × 0.5<br/>重新生成 ppo_override"]
    REDUCE_LR --> RETRY["status = pending<br/>用同一起始模型重试"]
    RETRY --> LAUNCH_RETRY([重新 _start_sub_phase])

    RETRY_CHECK -->|否| EXHAUSTED["rollback_exhausted<br/>继续推进 (不丢弃结果)"]
    EXHAUSTED --> RECORD

    ROLLBACK_CHECK -->|"否 (正常)"| RECORD["记录 stage_history<br/>更新 best_checkpoint_path<br/>starting_reward = best_reward"]

    RECORD --> POST["Post-Phase Pipeline"]
    EXHAUSTED --> POST

    POST --> JIT["_export_jit()<br/>checkpoint → policy.pt"]
    JIT --> MUJOCO["_record_mujoco_video()<br/>EGL 离屏渲染"]
    MUJOCO --> ISAAC["_record_video()<br/>Isaac Sim headless"]
    ISAAC --> PLOTS["_generate_plots()"]
    PLOTS --> PHASE_COMP["_check_phase_completion()<br/>是否是 phase 最后一个 sub_phase?"]

    PHASE_COMP -->|是| RECORD_PHASE["记录 phase_history"]
    PHASE_COMP -->|否| ADVANCE

    RECORD_PHASE --> ADVANCE["_advance()"]
    ADVANCE --> NEXT{"还有下一个<br/>sub_phase?"}
    NEXT -->|是| SET_NEXT["current_stage_id = next<br/>status = pending"]
    NEXT -->|否| DONE([全部完成])

    SET_NEXT --> NEXT_LOOP([回到主循环])

    style ROLLBACK_CHECK fill:#E74C3C,color:#fff
    style POST fill:#9B59B6,color:#fff
    style REDUCE_LR fill:#F39C12,color:#fff
```

## 7. Training Launch — Multi-GPU

TrainingLauncher 的多 GPU 训练启动流程：

```mermaid
flowchart LR
    subgraph "TrainingLauncher"
        BUILD["构建命令"]
        SINGLE["_build_single_cmd()<br/>python train.py<br/>--device cuda:0"]
        MULTI["_build_multigpu_cmd()<br/>torchrun --nproc_per_node=N<br/>train_multigpu.py<br/>--distributed"]
        EXEC["subprocess.Popen<br/>preexec_fn=os.setsid<br/>日志 > logs/train_<name>.log"]
        PID["返回 Popen 对象<br/>(含 pid)"]
    end

    BUILD -->|N=1| SINGLE
    BUILD -->|N>1| MULTI

    SINGLE --> EXEC
    MULTI --> EXEC
    EXEC --> PID

    subgraph "训练进程"
        ISAAC["Isaac Sim 初始化<br/>Omniverse Kit 加载"]
        ENV["创建 N × num_envs 个<br/>并行环境"]
        TRAIN_LOOP["rsl-rl PPO 训练循环<br/>每 iter: collect → learn → log"]
        TB_WRITE["写入 TensorBoard events"]
        CKPT_SAVE["每 save_interval<br/>保存 model_N.pt"]
    end

    PID --> ISAAC --> ENV --> TRAIN_LOOP
    TRAIN_LOOP --> TB_WRITE
    TRAIN_LOOP --> CKPT_SAVE

    subgraph "Graceful Stop"
        TERM["SIGTERM<br/>给进程组"]
        WAIT["等待 30s"]
        KILL["SIGKILL<br/>强制终止"]
    end

    style EXEC fill:#27AE60,color:#fff
    style TRAIN_LOOP fill:#4A90D9,color:#fff
```

## 8. Crash Recovery

崩溃恢复机制：

```mermaid
flowchart TD
    subgraph "每次状态变更后"
        SAVE["StateStore.save()<br/>原子写入 tempfile<br/>os.replace → orchestrator_state.json"]
    end

    subgraph "下次启动"
        LOAD["StateStore.load()"]
        LOAD --> HAS_STATE{有 state 文件?}
        HAS_STATE -->|无| NEW_RUN["初始化新 OrchestrationState"]
        HAS_STATE -->|有| CHECK_STATUS{current_stage_status?}

        CHECK_STATUS -->|running| CHECK_PID{PID 存活?}
        CHECK_PID -->|是| RESUME["_resume_monitor()<br/>重连 EmbeddedMonitor<br/>→ 进入主循环"]
        CHECK_PID -->|否| FAIL["status = failed<br/>→ 进入主循环"]

        CHECK_STATUS -->|pending| DIRECT["直接进入主循环<br/>→ _start_sub_phase"]
        CHECK_STATUS -->|complete| ADVANCE["_advance()"]
        CHECK_STATUS -->|failed| FAIL2["_handle_failure()<br/>retry 或退出"]
        CHECK_STATUS -->|overfitting| HANDLE["_handle_overfitting()"]
    end

    style SAVE fill:#8E44AD,color:#fff
    style LOAD fill:#8E44AD,color:#fff
```

## 9. 5-Phase Curriculum Pipeline

```mermaid
flowchart LR
    subgraph "p1: Flat Bootstrap"
        P1C["p1_coarse<br/>LR=1e-3<br/>no velocity tracking"]
        P1F["p1_fine<br/>LR=5e-4<br/>stand_still penalty"]
        P1C --> P1F
    end

    subgraph "p2: Flat Velocity"
        P2C["p2_coarse<br/>LR=1e-3<br/>vel [-0.3, 0.5]"]
        P2F["p2_fine<br/>LR=5e-4<br/>vel [-0.5, 1.0]"]
        P2C --> P2F
    end

    subgraph "p3: Gentle Terrain"
        P3C["p3_coarse<br/>LR=5e-4<br/>70% flat + 30% random_grid"]
        P3F["p3_fine<br/>LR=3e-4<br/>70% flat + 30% grid 0.5"]
        P3C --> P3F
    end

    subgraph "p3b: Intermediate"
        P3BC["p3b_coarse<br/>LR=1e-4<br/>50% flat + stairs + boxes"]
        P3BF["p3b_fine<br/>LR=8e-5<br/>difficulty 0.4-0.6"]
        P3BC --> P3BF
    end

    subgraph "p4: Rough Terrain"
        P4C["p4_coarse<br/>LR=1e-4<br/>30% flat + stairs/gap/boxes"]
        P4F["p4_fine<br/>LR=8e-5<br/>difficulty 0.5-0.7"]
        P4C --> P4F
    end

    subgraph "p5: Full Terrain"
        P5C["p5_coarse<br/>LR=1e-4<br/>20% each terrain type"]
        P5F["p5_fine<br/>LR=5e-5<br/>final polish"]
        P5C --> P5F
    end

    P1F -->|best ckpt| P2C
    P2F -->|best ckpt| P3C
    P3F -->|best ckpt| P3BC
    P3BF -->|best ckpt| P4C
    P4F -->|best ckpt| P5C
    P5F -->|final model| DONE(["部署"])

    style P3BC fill:#F39C12,color:#fff
    style P3BF fill:#F39C12,color:#fff
```

## 10. File Dependency Map

```mermaid
graph LR
    subgraph "automation/"
        __init__["__init__.py"]
        __main__["__main__.py"]
        PO["phase_orchestrator.py"]
        PM["phase_manager.py"]
        CG["config_generator.py"]
        PP["ppo_override.py"]
        EM["embedded_monitor.py"]
        TL["training_launcher.py"]
        SS["state_store.py"]
    end

    subgraph "外部依赖"
        TM["train_monitor.py"]
        YML["YAML Plan"]
        TEMPLATE["velocity_env_cfg.py<br/>(模板)"]
        TRAIN["train.py<br/>train_multigpu.py"]
    end

    __main__ --> PO
    PO --> PM & CG & PP & EM & TL & SS
    PM --> YML
    CG --> TEMPLATE
    TL --> TRAIN
    EM --> TM

    style PO fill:#4A90D9,color:#fff
```

## Module Summary

| File | LoC | Purpose | Used By |
|------|-----|---------|---------|
| `phase_orchestrator.py` | ~1150 | 两层级事件循环，全流程编排 | `__main__.py` |
| `phase_manager.py` | ~280 | YAML 解析，三层参数合并 | `phase_orchestrator.py` |
| `config_generator.py` | ~400 | 正则替换生成 env config | `phase_orchestrator.py` |
| `ppo_override.py` | ~130 | 生成 PPO 覆盖配置 | `phase_orchestrator.py` |
| `embedded_monitor.py` | ~230 | 增量监控 + 过拟合检测 | `phase_orchestrator.py` |
| `training_launcher.py` | ~180 | 子进程管理 (单/多 GPU) | `phase_orchestrator.py` |
| `state_store.py` | ~120 | 原子 JSON 状态持久化 | `phase_orchestrator.py` |
| `train_monitor.py` | ~700 | 独立监控工具，5 信号检测 | `embedded_monitor.py` |
