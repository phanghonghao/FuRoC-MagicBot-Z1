# Z1 Training Logging Architecture

## 日志来源总览

训练系统共产生 **4 类日志**，分别由不同层级的组件生成：

| 日志文件 | 位置 | 生成者 | 内容 |
|----------|------|--------|------|
| `train_<phase>.log` | `~/magiclab_rl_lab/logs/` | `training_launcher.py` | Isaac Sim 全部输出（GPU 初始化、env 创建、训练迭代） |
| `z1_pipeline_*.log` | `/tmp/` | `phase_orchestrator.py` | Pipeline 调度日志（phase 切换、超时、重试、错误） |
| `z1_monitor.log` | `/tmp/` | `train_monitor.py` | 过拟合检测、best model 识别 |
| `events.out.tfevents.*` | `logs/rsl_rl/.../<run_dir>/` | RSL-RL `OnPolicyRunner` | TensorBoard 指标数据 |

## 日志流程图

```mermaid
flowchart TB
    subgraph SSH["SSH 启动层 (本地 → RTX)"]
        CMD["nohup python phase_orchestrator.py<br/>> /tmp/z1_pipeline_p3b.log 2>&1 &"]
        MON_CMD["nohup python train_monitor.py<br/>--poll_interval 120<br/>> /tmp/z1_monitor.log 2>&1 &"]
    end

    subgraph ORCH["phase_orchestrator.py<br/>(Pipeline 调度器)"]
        direction TB
        PM[phase_manager.py<br/>YAML 解析]
        CG[config_generator.py<br/>环境配置生成]
        PO[ppo_override.py<br/>PPO 参数覆盖]
        TL["training_launcher.py<br/>子进程管理"]
        EM[embedded_monitor.py<br/>过拟合检测]
        SS[state_store.py<br/>状态持久化]

        PM --> CG --> PO --> TL
        TL --> EM --> SS
    end

    subgraph LAUNCH["training_launcher.py<br/>(训练启动器)"]
        direction TB
        BUILD["构建 torchrun 命令"]
        REDIR["打开 train_&lt;phase&gt;.log<br/>stdout+stderr 重定向到文件"]
        POPEN["subprocess.Popen(...,<br/>stdout=fd, stderr=STDOUT)"]
        BUILD --> REDIR --> POPEN
    end

    subgraph TORCH["torchrun (4 GPU workers)"]
        RANK0["RANK 0 (cuda:0)<br/>创建 run directory"]
        RANK1["RANK 1 (cuda:1)"]
        RANK2["RANK 2 (cuda:2)"]
        RANK3["RANK 3 (cuda:3)"]
    end

    subgraph ISAAC["Isaac Sim / train.py"]
        direction TB
        INIT["Omniverse Kit 初始化<br/>GPU 枚举 · KVDB 锁 · EGL"]
        ENV["环境创建<br/>terrain · robot · curriculum"]
        RUNNER["OnPolicyRunner 初始化<br/>PPO · Actor-Critic 网络"]
        TRAIN["训练循环<br/>Learning iteration N/M"]
        INIT --> ENV --> RUNNER --> TRAIN
    end

    subgraph RSLRL["RSL-RL OnPolicyRunner"]
        TB_LOG["TensorBoard Writer<br/>写入 events.out.tfevents.*"]
        CONSOLE["控制台输出<br/>reward · ep_len · time_out · entropy"]
    end

    subgraph LOGS["日志文件 (RTX 服务器)"]
        direction LR
        L1["/tmp/z1_pipeline_p3b.log<br/>Pipeline 调度日志<br/><b>来源:</b> nohup 重定向"]
        L2["~/magiclab_rl_lab/logs/train_p3b_fine.log<br/>训练详细日志<br/><b>来源:</b> training_launcher 重定向"]
        L3["/tmp/z1_monitor.log<br/>监控日志<br/><b>来源:</b> nohup 重定向"]
        L4["logs/rsl_rl/.../events.out.tfevents.*<br/>TensorBoard 数据<br/><b>来源:</b> RSL-RL 内置"]
    end

    CMD --> ORCH
    MON_CMD --> LOGS

    ORCH -->|"每个 sub-phase"| LAUNCH
    LAUNCH -->|"启动子进程"| TORCH
    TORCH -->|"每个 rank"| ISAAC

    ISAAC -->|"训练输出"| RSLRL
    RSLRL -->|"Python logging"| CONSOLE
    RSLRL -->|"TB writer"| TB_LOG
    CONSOLE -->|"stdout → fd"| L2
    TB_LOG -->|"文件写入"| L4

    ORCH -->|"Python logging"| L1
    EM -->|"TensorBoard 轮询"| L4

    style L1 fill:#ffe0b2
    style L2 fill:#c8e6c9
    style L3 fill:#bbdefb
    style L4 fill:#e1bee7
    style INIT fill:#ffcdd2
```

## 各日志详解

### 1. Pipeline 日志 — `train_<phase>.log`

**生成方式：** `training_launcher.py` 用 `subprocess.Popen` 启动 torchrun，将 stdout/stderr 重定向到文件。

```
training_launcher.py 第 58-81 行:
  log_file = self._log_dir / f"train_{run_name}.log"
  fd = open(log_file, "w")
  proc = subprocess.Popen(cmd, stdout=fd, stderr=subprocess.STDOUT)
```

**内容：** Isaac Sim 完整启动日志，包括：
- GPU 枚举与 Vulkan 信息
- Omniverse Kit 加载、KVDB 锁状态
- 环境创建（terrain、robot、curriculum）
- Actor/Critic 网络结构
- RSL-RL 训练迭代（`Learning iteration N/M`）

### 2. Orchestrator 日志 — `z1_pipeline_*.log`

**生成方式：** 本地 SSH 启动 orchestrator 时通过 nohup 重定向。

```bash
ssh ... "nohup python phase_orchestrator.py ... > /tmp/z1_pipeline_p3b.log 2>&1 &"
```

**内容：** Pipeline 调度层面的事件：
- Sub-phase 启动/停止
- Run directory 查找（每 30s 轮询，最多 8 次）
- 过拟合检测结果
- 重试与回滚决策
- 错误与终止原因

### 3. Monitor 日志 — `z1_monitor.log`

**生成方式：** 同样通过 nohup 重定向。

```bash
ssh ... "nohup python train_monitor.py ... > /tmp/z1_monitor.log 2>&1 &"
```

**内容：** 所有 run 的定期扫描结果：
- 每 120s 扫描所有 run 目录
- 读取最新 checkpoint 的 TensorBoard 数据
- 过拟合检测（reward 下降、entropy 坍缩等）
- 最佳 checkpoint 标记

### 4. TensorBoard 数据 — `events.out.tfevents.*`

**生成方式：** RSL-RL 库内置的 TensorBoard writer，由 `OnPolicyRunner` 自动创建。

**内容：** 结构化的训练指标：
- Episode Reward（各项细分）
- Episode Termination（time_out、bad_orientation）
- Curriculum（terrain_levels、vel_levels）
- Loss（entropy、value_loss）

## p3b_fine 失败案例复盘

```mermaid
sequenceDiagram
    participant ORCH as phase_orchestrator
    participant TL as training_launcher
    participant KIT as Isaac Sim (RANK 0)
    participant FS as Run Directory

    Note over ORCH: p3b_coarse 过拟合检测 → 杀掉训练
    ORCH->>TL: 立即启动 p3b_fine（未等待 Kit 退出）
    TL->>KIT: torchrun 启动 4 workers
    KIT->>KIT: Omniverse Kit 初始化
    Note over KIT: ⚠️ KVDB locked by previous Kit process
    Note over KIT: Disabling key-value database
    KIT->>KIT: RANK 1/2/3 环境创建成功
    KIT->>KIT: RANK 0 卡在 asset download
    Note over KIT: Waiting for server response... 30s...60s...90s...120s
    Note over FS: ❌ Run directory 未创建

    loop 每 30s × 8 次
        ORCH->>FS: 检查 run directory → 不存在
    end
    Note over ORCH: 4 分钟超时
    ORCH->>TL: SIGTERM 杀掉进程
    TL->>KIT: SIGTERM → SIGKILL

    Note over ORCH: 重试第 2 次 → 同样失败
    Note over ORCH: Pipeline 停止
```

**根因：** Orchestrator 在杀掉 p3b_coarse 训练后**未等待 Kit 进程完全退出**就立即启动 p3b_fine，导致 KVDB 锁冲突。
