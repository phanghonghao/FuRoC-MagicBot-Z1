#!/bin/bash
# ============================================================
# watch_gpu - GPU 占用监控，检测绕过 Slurm 的直接占用
# 部署位置: /usr/local/bin/watch_gpu (所有用户可用)
# 用法:
#   watch_gpu              单次输出到 stdout
#   watch_gpu -w 5         每5秒刷新
#   watch_gpu --log        单次输出 + 追加日志到 /var/log/slurm/watch_gpu.log
#   watch_gpu --cron       静默模式，只记录违规条目到日志（用于 crontab）
# ============================================================

LOG_FILE="/var/log/slurm/watch_gpu.log"
LOG_MAX_SIZE=$((10 * 1024 * 1024))  # 10MB

MODE="stdout"
REFRESH=""

# --- 参数解析 ---
while [ $# -gt 0 ]; do
    case "$1" in
        -w)
            if [ -n "$2" ]; then
                REFRESH="$2"
                shift 2
            else
                echo "Usage: watch_gpu -w <seconds>" >&2
                exit 1
            fi
            ;;
        --log)
            MODE="log"
            shift
            ;;
        --cron)
            MODE="cron"
            shift
            ;;
        *)
            echo "Usage: watch_gpu [-w <seconds>] [--log] [--cron]" >&2
            exit 1
            ;;
    esac
done

# --- 日志轮转 ---
rotate_log() {
    if [ -f "$LOG_FILE" ]; then
        local size
        size=$(stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)
        if [ "$size" -gt "$LOG_MAX_SIZE" ]; then
            local half=$((size / 2))
            tail -c "$half" "$LOG_FILE" > "${LOG_FILE}.tmp" 2>/dev/null
            mv -f "${LOG_FILE}.tmp" "$LOG_FILE" 2>/dev/null
        fi
    fi
}

# --- 写入日志 ---
log_line() {
    local line="$1"
    rotate_log
    echo "$line" >> "$LOG_FILE"
}

# --- 收集 Slurm 作业 PID ---
collect_slurm_pids() {
    SLURM_PIDS=""
    local job_ids
    job_ids=$(squeue -h -o "%i" 2>/dev/null)
    if [ -n "$job_ids" ]; then
        SLURM_PIDS=$(echo "$job_ids" | while read -r jid; do
            scontrol listpids "$jid" 2>/dev/null | tail -n +2 | awk '{print $1}'
        done | sort -u | tr '\n' ' ')
    fi
}

# --- 检查单个 PID 是否来自 Slurm ---
check_slurm_pid() {
    local pid="$1"
    # 方法1: 检查已知 Slurm PID 列表
    if [ -n "$SLURM_PIDS" ]; then
        for spid in $SLURM_PIDS; do
            if [ "$pid" = "$spid" ]; then
                echo "slurm"
                return
            fi
        done
    fi
    # 方法2: 检查 /proc/$pid/environ 中是否有 SLURM_JOB_ID
    local sjob
    sjob=$(cat /proc/"$pid"/environ 2>/dev/null | tr '\0' '\n' | grep "^SLURM_JOB_ID=" | cut -d= -f2)
    if [ -n "$sjob" ]; then
        echo "slurm:$sjob"
        return
    fi
    echo "bypass"
}

# --- 生成完整报告 ---
run_report() {
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    local SHOW="no"
    if [ "$MODE" = "stdout" ] || [ "$MODE" = "log" ]; then
        SHOW="yes"
        # 只在交互式终端 clear
        if [ -t 1 ]; then
            clear 2>/dev/null || true
        fi
        echo "============================================"
        echo " GPU Monitor - pro6000d ($timestamp)"
        echo "============================================"
    fi

    # --- GPU 概览 ---
    if [ "$SHOW" = "yes" ]; then
        echo ""
        echo "--- GPU Status ---"
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null | \
            while IFS=',' read -r idx mem_used mem_total util temp; do
                mem_used=$(echo "$mem_used" | tr -d ' ')
                mem_total=$(echo "$mem_total" | tr -d ' ')
                util=$(echo "$util" | tr -d ' ')
                temp=$(echo "$temp" | tr -d ' ')
                printf "  GPU %s: %s / %s | Util: %s | Temp: %s\n" "$idx" "$mem_used" "$mem_total" "$util" "$temp"
            done
    fi

    # --- Slurm 作业 ---
    collect_slurm_pids

    if [ "$SHOW" = "yes" ]; then
        echo ""
        echo "--- Slurm Jobs ---"
        local jobs
        jobs=$(squeue -h -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R" 2>/dev/null)
        if [ -z "$jobs" ]; then
            echo "  (no running jobs)"
        else
            echo "$jobs" | while IFS= read -r line; do
                echo "  $line"
            done
        fi
    fi

    # --- GPU 进程详情 + 绕过检测 ---
    if [ "$SHOW" = "yes" ]; then
        echo ""
        echo "--- GPU Processes ---"
    fi

    FOUND_PROC=0
    BYPASS_COUNT=0

    for i in $(seq 0 7); do
        PROCS=$(nvidia-smi -i "$i" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
        if [ -n "$PROCS" ]; then
            FOUND_PROC=1
            while IFS=',' read -r pid pname mem; do
                pid=$(echo "$pid" | tr -d ' ')
                pname=$(echo "$pname" | tr -d ' ')
                mem=$(echo "$mem" | tr -d ' ')
                user=$(ps -o user= -p "$pid" 2>/dev/null || echo "?")
                start=$(ps -o lstart= -p "$pid" 2>/dev/null || echo "?")
                cmd=$(ps -o args= -p "$pid" 2>/dev/null | cut -c1-80 || echo "?")

                # 检查是否经 Slurm 分配
                local check
                check=$(check_slurm_pid "$pid")

                local SLURM_FLAG=""
                case "$check" in
                    slurm)      SLURM_FLAG="[Slurm]" ;;
                    slurm:*)    SLURM_FLAG="[Slurm:Job#${check#slurm:}]" ;;
                    bypass)     SLURM_FLAG="[!! BYPASS !!]"; BYPASS_COUNT=$((BYPASS_COUNT + 1)) ;;
                esac

                # stdout/log 模式: 打印完整信息
                if [ "$SHOW" = "yes" ]; then
                    printf "  GPU %s | PID %-8s | %-10s | %8s | %-8s | %s %s\n" \
                        "$i" "$pid" "$user" "$mem" "$pname" "$SLURM_FLAG" "$start"
                fi

                # log/cron 模式: 写入结构化日志
                if [ "$MODE" = "log" ] || [ "$MODE" = "cron" ]; then
                    local tag
                    case "$check" in
                        bypass) tag="BYPASS" ;;
                        *)      tag="SLURM" ;;
                    esac
                    log_line "$timestamp | GPU $i | PID $pid | $user | $tag | $cmd"
                fi
            done <<< "$PROCS"
        fi
    done

    if [ "$SHOW" = "yes" ]; then
        if [ "$FOUND_PROC" = "0" ]; then
            echo "  (all GPUs idle)"
        fi

        # --- 违规汇总 ---
        echo ""
        echo "--- Bypass Detection ---"
        if [ "$BYPASS_COUNT" = "0" ]; then
            echo "  All GPU processes are using Slurm."
        else
            # 重新遍历显示违规条目（BYPASS_COUNT 已在上面统计）
            for i in $(seq 0 7); do
                PROCS=$(nvidia-smi -i "$i" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
                if [ -n "$PROCS" ]; then
                    for pid in $PROCS; do
                        local check2
                        check2=$(check_slurm_pid "$pid")
                        if [ "$check2" = "bypass" ]; then
                            user=$(ps -o user= -p "$pid" 2>/dev/null || echo "?")
                            printf "  !! GPU %s | PID %s | User: %s | NOT using Slurm\n" "$i" "$pid" "$user"
                        fi
                    done
                fi
            done
        fi

        echo ""
        echo "============================================"
    fi
}

# --- 主循环 ---
if [ -n "$REFRESH" ]; then
    while true; do
        run_report
        sleep "$REFRESH"
    done
else
    run_report
fi
