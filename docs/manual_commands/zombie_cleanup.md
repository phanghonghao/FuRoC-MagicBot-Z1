# 僵尸进程清理

## 常见僵尸进程

### Omniverse Kit 残留（导致 KVDB 锁冲突）

```bash
# 查找
pgrep -f "omni.kit|kit_" | while read pid; do ps -o user=,pid=,etime=,cmd= -p $pid; done

# 清理
pkill -f "omni.kit|kit_"
```

### 训练 Monitor 残留

```bash
ps aux | grep train_monitor | grep -v grep
# kill <PID>
```

### Telemetry 进程（占用 GPU 显存）

```bash
ps aux | grep telemetry | grep -v grep
# kill -9 <PID>
```

## 全面清理脚本

一键清理所有可能的僵尸进程：

```bash
echo "=== 僵尸进程扫描 ==="

# 1. Kit 残留
KIT_PIDS=$(pgrep -f "omni.kit|kit_" 2>/dev/null)
if [ -n "$KIT_PIDS" ]; then
    echo "[KIT] 发现: $(echo $KIT_PIDS | tr '\n' ' ')"
    echo "$KIT_PIDS" | xargs kill -9 2>/dev/null
    echo "[KIT] 已清理"
else
    echo "[KIT] 无"
fi

# 2. 旧训练进程
TRAIN_PIDS=$(ps aux | grep -E "train.py|train_multigpu|torchrun" | grep phh | grep -v grep | awk '{print $2}')
if [ -n "$TRAIN_PIDS" ]; then
    echo "[TRAIN] 发现: $(echo $TRAIN_PIDS | tr '\n' ' ')"
    echo "$TRAIN_PIDS" | xargs kill -9 2>/dev/null
    echo "[TRAIN] 已清理"
else
    echo "[TRAIN] 无"
fi

# 3. Monitor
MONITOR_PIDS=$(ps aux | grep train_monitor | grep -v grep | awk '{print $2}')
if [ -n "$MONITOR_PIDS" ]; then
    echo "[MONITOR] 发现: $(echo $MONITOR_PIDS | tr '\n' ' ')"
    echo "$MONITOR_PIDS" | xargs kill -9 2>/dev/null
    echo "[MONITOR] 已清理"
else
    echo "[MONITOR] 无"
fi

# 4. Telemetry
TELE_PIDS=$(ps aux | grep "omni.telemetry" | grep -v grep | awk '{print $2}')
if [ -n "$TELE_PIDS" ]; then
    echo "[TELEMETRY] 发现: $(echo $TELE_PIDS | tr '\n' ' ')"
    echo "$TELE_PIDS" | xargs kill -9 2>/dev/null
    echo "[TELEMETRY] 已清理"
else
    echo "[TELEMETRY] 无"
fi

# 5. Orchestrator
ORCH_PIDS=$(ps aux | grep phase_orchestrator | grep python | grep -v grep | awk '{print $2}')
if [ -n "$ORCH_PIDS" ]; then
    echo "[ORCHESTRATOR] 发现: $(echo $ORCH_PIDS | tr '\n' ' ')"
    echo "$ORCH_PIDS" | xargs kill -9 2>/dev/null
    echo "[ORCHESTRATOR] 已清理"
else
    echo "[ORCHESTRATOR] 无"
fi

# 6. __pycache__
find ~/magiclab_rl_lab/scripts/automation -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
echo "[CACHE] automation __pycache__ 已清理"

sleep 2
echo ""
echo "=== GPU 状态 ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
```

## 按 GPU 显存排查

```bash
# 查看每个 GPU 上的进程
nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader 2>/dev/null | while IFS="," read bus pid name mem; do
    pid=$(echo $pid | tr -d " ")
    user=$(ps -o user= -p $pid 2>/dev/null)
    echo "PID=$pid user=$user gpu_bus=$bus $name $mem"
done
```
