#!/bin/bash
# ============================================================
# Slurm 单节点安装脚本 - RTX Pro 6000D 服务器 (pro6000d)
# 8x RTX 6000D (85GB each), 256 CPUs, 1.5TB RAM
# 适用于: Ubuntu 22.04 LTS (cgroup v2 混合模式)
# 使用方法: sudo bash install_slurm.sh
# 状态: 已验证通过 (2026-05-13)
# ============================================================

set -e

CLUSTER_NAME="pro6000d-cluster"
HOSTNAME="pro6000d"
SLURM_USER="slurm"
SLURM_CTLD_PORT=6817
SLURM_D_PORT=6818
SLURM_CTLD_LOG="/var/log/slurm/slurmctld.log"
SLURM_D_LOG="/var/log/slurm/slurmd.log"

echo "========================================"
echo " Slurm 安装脚本 - ${HOSTNAME}"
echo "========================================"

# ----------------------------------------------------------
# Step 1: 安装依赖和 MUNGE
# ----------------------------------------------------------
echo ""
echo "[Step 1/7] 安装依赖包和 MUNGE..."
apt-get update
apt-get install -y munge libmunge-dev libmunge2 slurm-wlm slurm-wlm-basic-plugins slurmctld slurmd slurm-client hwloc

echo "  MUNGE + Slurm 包安装完成"

# ----------------------------------------------------------
# Step 2: 配置 MUNGE 认证
# ----------------------------------------------------------
echo ""
echo "[Step 2/7] 配置 MUNGE 认证..."

mkdir -p /etc/munge /var/lib/munge /var/log/munge /run/munge
chown -R munge:munge /etc/munge /var/lib/munge /var/log/munge /run/munge

if [ ! -f /etc/munge/munge.key ]; then
    mungekey -f --create
    echo "  MUNGE key 已生成"
else
    echo "  MUNGE key 已存在，跳过"
fi

chmod 400 /etc/munge/munge.key
chown munge:munge /etc/munge/munge.key

systemctl enable munge
systemctl restart munge
sleep 1

if systemctl is-active --quiet munge; then
    echo "  MUNGE 服务启动成功"
else
    echo "  [ERROR] MUNGE 启动失败!"
    exit 1
fi

# ----------------------------------------------------------
# Step 3: 创建 Slurm 用户和目录
# ----------------------------------------------------------
echo ""
echo "[Step 3/7] 创建 Slurm 用户和目录..."
id -u ${SLURM_USER} >/dev/null 2>&1 || useradd -r -s /usr/sbin/nologin ${SLURM_USER}

mkdir -p /etc/slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm /var/run/slurm
chown -R ${SLURM_USER}:${SLURM_USER} /var/spool/slurmctld /var/spool/slurmd /var/log/slurm /var/run/slurm
echo "  目录创建完成"

# ----------------------------------------------------------
# Step 4: 写入 slurm.conf
# 注意: Ubuntu 22.04 Slurm 21.08.5 兼容性:
#   - PidFile 不是合法指令 (用 SlurmctldPidFile/SlurmdPidFile)
#   - AccountingStorageLoc 已被移除
#   - accounting_storage/filetxt 不可用, 用 none
#   - cgroup v2 下 task/cgroup 不可用, 用 task/none
#   - proctrack/cgroup 的 cpuset 不在 v1, 用 proctrack/linuxproc
# ----------------------------------------------------------
echo ""
echo "[Step 4/7] 写入 /etc/slurm/slurm.conf ..."

REAL_CPUS=$(nproc)
REAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
REAL_MEM_MB=$((REAL_MEM_KB / 1024))

cat > /etc/slurm/slurm.conf << SLURMCONF
# ============================================================
# Slurm 配置 - pro6000d 单节点 8-GPU
# 由 install_slurm.sh 自动生成
# ============================================================

ClusterName=${CLUSTER_NAME}
SlurmctldHost=${HOSTNAME}
SlurmctldPort=${SLURM_CTLD_PORT}
SlurmdPort=${SLURM_D_PORT}

AuthType=auth/munge
SlurmUser=${SLURM_USER}

StateSaveLocation=/var/spool/slurmctld
SlurmdSpoolDir=/var/spool/slurmd
SlurmdPidFile=/var/run/slurmd.pid
SlurmctldPidFile=/var/run/slurmctld.pid

SlurmctldLogFile=${SLURM_CTLD_LOG}
SlurmdLogFile=${SLURM_D_LOG}
SlurmctldDebug=info
SlurmdDebug=info

PluginDir=/usr/lib/x86_64-linux-gnu/slurm-wlm
ProctrackType=proctrack/linuxproc
TaskPlugin=task/none
GresTypes=gpu

SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core

PriorityType=priority/multifactor
PriorityWeightAge=1000
PriorityWeightFairshare=10000
PriorityWeightJobSize=1000
PriorityWeightQOS=10000
PriorityDecayHalfLife=7-0

AccountingStorageType=accounting_storage/none
JobCompType=jobcomp/filetxt
JobCompLoc=/var/log/slurm/jobcomp

SlurmctldTimeout=120
SlurmdTimeout=300
InactiveLimit=0
MinJobAge=300
WaitTime=0

ReturnToService=2
MaxArraySize=1000
MaxJobCount=10000

# --- 节点 ---
NodeName=${HOSTNAME} CPUs=${REAL_CPUS} RealMemory=${REAL_MEM_MB} Gres=gpu:8 State=UNKNOWN

# --- 分区 ---
PartitionName=gpu Nodes=${HOSTNAME} Default=YES MaxTime=INFINITE State=UP DefaultTime=01:00:00
PartitionName=cpu Nodes=${HOSTNAME} Default=NO MaxTime=INFINITE State=UP OverSubscribe=YES
SLURMCONF

echo "  slurm.conf 已写入 (${REAL_CPUS} CPUs, ${REAL_MEM_MB}MB RAM, 8 GPUs)"

# ----------------------------------------------------------
# Step 5: 写入 gres.conf (GPU 资源)
# ----------------------------------------------------------
echo ""
echo "[Step 5/7] 写入 /etc/slurm/gres.conf ..."

cat > /etc/slurm/gres.conf << 'GRESEOF'
# GPU Generic Resources - 8x RTX 6000D
GRESEOF

for i in $(seq 0 7); do
    if [ -e "/dev/nvidia${i}" ]; then
        echo "NodeName=${HOSTNAME} Name=gpu File=/dev/nvidia${i}" >> /etc/slurm/gres.conf
        echo "  GPU $i: OK"
    else
        echo "NodeName=${HOSTNAME} Name=gpu File=/dev/nvidia${i}" >> /etc/slurm/gres.conf
        echo "  GPU $i: device not found (added anyway)"
    fi
done

# ----------------------------------------------------------
# Step 6: 写入 cgroup.conf (最小化配置, v2 兼容)
# ----------------------------------------------------------
echo ""
echo "[Step 6/7] 写入 cgroup.conf ..."

cat > /etc/slurm/cgroup.conf << 'CGEOF'
CgroupMountpoint=/sys/fs/cgroup
CgroupAutomount=yes
CGEOF

echo "  cgroup.conf 已写入"

# ----------------------------------------------------------
# Step 7: 启动 Slurm 服务
# ----------------------------------------------------------
echo ""
echo "[Step 7/7] 启动 Slurm 服务..."

systemctl enable slurmctld slurmd
systemctl restart slurmctld
sleep 2
systemctl restart slurmd
sleep 2

echo ""
echo "=== 服务状态 ==="
echo "  slurmctld: $(systemctl is-active slurmctld)"
echo "  slurmd:    $(systemctl is-active slurmd)"

echo ""
echo "========================================"
echo " 安装完成!"
echo "========================================"
echo ""
echo " 验证:  sinfo && srun --gpus=1 nvidia-smi"
echo ""
echo " 常用命令:"
echo "   sinfo                          # 查看分区/节点状态"
echo "   squeue -u \$USER                # 查看你的作业"
echo "   srun --gpus=1 -c 8 python train.py   # 交互式申请1个GPU"
echo "   srun --gpus=4 -c 16 torchrun --nproc_per_node=4 train.py"
echo "   sbatch job.sh                  # 提交批处理脚本"
echo "   scancel <jobid>                # 取消作业"
echo ""
