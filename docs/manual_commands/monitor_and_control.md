# 监控与控制

## 查看训练状态

```bash
# 训练进程是否在跑
ps aux | grep -E "train.py|train_multigpu|torchrun" | grep phh | grep -v grep

# Orchestrator 进程
ps aux | grep phase_orchestrator | grep python | grep -v grep

# Orchestrator 状态
cat ~/magiclab_rl_lab/orchestrator_state.json | python -m json.tool
```

## 查看训练日志

```bash
# 最新 30 行
tail -30 /tmp/z1_p3b_coarse.log

# 实时跟踪
tail -f /tmp/z1_p3b_coarse.log

# Orchestrator 日志
tail -40 /tmp/z1_5phase_pipeline.log

# 查看所有训练日志文件
ls -lht /tmp/z1_*.log
```

## 查看 GPU 状态

```bash
# 完整 GPU 信息
nvidia-smi

# 简洁版
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits

# 只看自己的进程（用户 phh）
nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader 2>/dev/null
```

## 查看保存的模型

```bash
# 所有 run 的 model
ls -lht logs/rsl_rl/magiclab_z1_12dof_velocity/*/model_*.pt | head -20

# 指定 run 的 model
ls -lht logs/rsl_rl/magiclab_z1_12dof_velocity/2026-05-09_*/model_*.pt
```

## 停止训练

```bash
# 找到 PID
ps aux | grep train | grep phh | grep -v grep

# 优雅停止
kill <PID>

# 强制停止（如果 kill 不掉）
kill -9 <PID>

# 停止整个 torchrun 进程组（包括所有 GPU worker）
kill -9 -$(ps -o pgid= -p <PID> | tr -d ' ')

# 停止 orchestrator
ps aux | grep phase_orchestrator | grep python | grep -v grep | awk '{print $2}' | xargs kill
```

## 停止后清理

```bash
# 清理残留 Kit 进程（防止 KVDB 锁冲突）
pgrep -f "omni.kit|kit_" | xargs kill -9 2>/dev/null

# 清理 __pycache__
find ~/magiclab_rl_lab/scripts/automation -name __pycache__ -exec rm -rf {} + 2>/dev/null
```
