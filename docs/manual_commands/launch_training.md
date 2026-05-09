# 启动训练

## 前置检查

```bash
# 检查是否有残留训练进程
ps aux | grep -E "train.py|train_multigpu|phase_orchestrator" | grep -v grep

# 如果有，先杀掉
kill <PID>

# 检查 GPU 空闲状态
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits

# 检查 EGL（Isaac Sim 必需）
test -f ~/miniconda3/envs/isaaclab/share/glvnd/egl_vendor.d/10_nvidia.json && echo "OK" || echo "MISSING"
# 如果 MISSING:
# cp /usr/share/glvnd/egl_vendor.d/10_nvidia.json ~/miniconda3/envs/isaaclab/share/glvnd/egl_vendor.d/
```

---

## 方式 A：直接 torchrun（用当前 velocity_env_cfg.py 的配置）

适合：不需要改地形/reward 参数，只是继续训练或换个 PPO 参数。

```bash
# 从某个 checkpoint 继续
CKPT_RUN="2026-05-08_07-09-14_p3_coarse_v2"
CKPT_MODEL="model_10500.pt"
RUN_NAME="p3b_coarse_manual"
MAX_ITER=15000
NUM_ENVS=4096

nohup torchrun \
    --nproc_per_node=4 \
    --master_port=29502 \
    scripts/rsl_rl/train_multigpu.py \
    --task=Magiclab-Z1-12dof-Velocity \
    --run_name=${RUN_NAME} \
    --headless \
    --distributed \
    --num_envs=${NUM_ENVS} \
    --max_iterations=${MAX_ITER} \
    --resume \
    --load_run=${CKPT_RUN} \
    --checkpoint=${CKPT_MODEL} \
    > /tmp/z1_${RUN_NAME}.log 2>&1 & echo PID=$!
```

如果需要自定义 PPO 参数（LR、entropy），加 `--agent_cfg`：

```bash
    --agent_cfg=tmp/phase_configs/p3b_coarse/ppo_override_cfg.py \
```

---

## 方式 B：先生成配置再启动（推荐，完整控制地形+reward）

### Step 1：生成配置

```bash
# orchestrator dry-run 只生成配置文件，不启动训练
python -u scripts/automation/phase_orchestrator.py \
    --plan training_plans/z1_5phase_plan.yaml \
    --start-from p3b_coarse \
    --fresh \
    --dry-run
```

生成文件位置：
- `tmp/phase_configs/p3b_coarse/velocity_env_cfg.py` — 地形 + reward
- `tmp/phase_configs/p3b_coarse/ppo_override_cfg.py` — PPO 参数

dry-run 会自动把 env config 覆盖到 `source/.../velocity_env_cfg.py`。

### Step 2：启动训练

```bash
CKPT_RUN="2026-05-08_07-09-14_p3_coarse_v2"
CKPT_MODEL="model_10500.pt"
RUN_NAME="p3b_coarse"
PHASE="p3b_coarse"

nohup torchrun \
    --nproc_per_node=4 \
    --master_port=29502 \
    scripts/rsl_rl/train_multigpu.py \
    --task=Magiclab-Z1-12dof-Velocity \
    --run_name=${RUN_NAME} \
    --headless \
    --distributed \
    --num_envs=4096 \
    --max_iterations=15000 \
    --resume \
    --load_run=${CKPT_RUN} \
    --checkpoint=${CKPT_MODEL} \
    --agent_cfg=tmp/phase_configs/${PHASE}/ppo_override_cfg.py \
    > /tmp/z1_${RUN_NAME}.log 2>&1 & echo PID=$!
```

---

## 方式 C：用 Orchestrator（全自动）

自动处理配置生成、过拟合检测、阶段推进、录视频。

```bash
# 清理旧状态
rm -f orchestrator_state.json

# 启动（从 p3b_coarse 开始）
nohup python -u scripts/automation/phase_orchestrator.py \
    --plan training_plans/z1_5phase_plan.yaml \
    --num-gpus 4 \
    --fresh \
    --start-from p3b_coarse \
    --poll-interval 120 \
    > /tmp/z1_5phase_pipeline.log 2>&1 & echo PID=$!

# 查看日志
tail -f /tmp/z1_5phase_pipeline.log
```

---

## 单卡训练

```bash
CKPT_RUN="2026-05-08_07-09-14_p3_coarse_v2"
CKPT_MODEL="model_10500.pt"
RUN_NAME="p3b_coarse_1gpu"

nohup python -u scripts/rsl_rl/train.py \
    --task Magiclab-Z1-12dof-Velocity \
    --run_name ${RUN_NAME} \
    --headless \
    --device cuda:0 \
    --num_envs 4096 \
    --max_iterations 15000 \
    --load_run ${CKPT_RUN} \
    --checkpoint ${CKPT_MODEL} \
    > /tmp/z1_${RUN_NAME}.log 2>&1 & echo PID=$!
```

---

## 不同 GPU 数量对应参数

| GPU 数 | --nproc_per_node | --num_envs (推荐) | 预估速度 |
|--------|-----------------|-------------------|---------|
| 1 | (用 train.py) | 4096 | ~25k steps/s |
| 2 | 2 | 8192 | ~50k steps/s |
| 4 | 4 | 16384 | ~100k steps/s |
| 8 | 8 | 32768 | ~200k steps/s |

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 进程启动后立刻死掉 | GPU 被其他进程占用 | `nvidia-smi` 检查，kill 僵尸进程 |
| 卡在初始化 10+ 分钟 | Isaac Sim 多 GPU 正常现象 | 耐心等待 |
| `KVDB lock` 错误 | 上一个 Kit 进程没退干净 | `pkill -f omni.kit` 再重试 |
| `.pyc` 缓存导致代码不更新 | Python 加载旧字节码 | `find . -name __pycache__ -exec rm -rf {} +` |
| `master_port` 冲突 | 另一个 torchrun 占用端口 | 改 `--master_port=29503` 或其他 |
