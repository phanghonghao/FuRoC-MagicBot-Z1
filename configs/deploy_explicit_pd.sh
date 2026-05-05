#!/bin/bash
# deploy_explicit_pd.sh — One-click deploy s5 explicit PD training to RTX server
set -euo pipefail

REMOTE="phh@192.168.120.155"
REMOTE_PROJECT="~/magiclab_rl_lab"
REMOTE_CONDA_ENV="isaaclab"

# Local paths (relative to this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT="$SCRIPT_DIR/../magiclab_rl_lab"

# Remote sub-paths for files to sync
ACTUATOR_REL="source/magiclab_rl_lab/magiclab_rl_lab/assets/robots/magiclab.py"
ENVCFG_REL="source/magiclab_rl_lab/magiclab_rl_lab/tasks/locomotion/robots/z1/12dof/velocity_env_cfg.py"

RUN_NAME="s5_explicit_pd"
LOG_FILE="/tmp/z1_mgpu_s5_explicit_pd.log"
NUM_GPUS=4
MASTER_PORT=29500
NUM_ENVS=4096
MAX_ITER=50000

echo "=========================================="
echo " Deploy s5 Explicit PD — 4 GPU Training"
echo "=========================================="
echo ""

# Step 1: Sync modified files to RTX
echo "[Step 1/5] Syncing modified files..."
scp "$LOCAL_PROJECT/$ACTUATOR_REL" "$REMOTE:$REMOTE_PROJECT/$ACTUATOR_REL"
scp "$LOCAL_PROJECT/$ENVCFG_REL" "$REMOTE:$REMOTE_PROJECT/$ENVCFG_REL"
echo "  Done."
echo ""

# Step 2: Re-install magiclab_rl_lab on RTX (so actuator changes take effect)
echo "[Step 2/5] Re-installing magiclab_rl_lab on RTX..."
ssh "$REMOTE" bash -s <<REMOTE_INSTALL
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate $REMOTE_CONDA_ENV
  cd $REMOTE_PROJECT
  pip install -e source/magiclab_rl_lab --quiet 2>&1 | tail -3
REMOTE_INSTALL
echo "  Done."
echo ""

# Step 3: Check EGL fix
echo "[Step 3/5] Checking EGL fix..."
if ssh "$REMOTE" "test -f ~/miniconda3/envs/$REMOTE_CONDA_ENV/share/glvnd/egl_vendor.d/10_nvidia.json"; then
    echo "  EGL fix OK."
else
    echo "  WARNING: EGL vendor file missing! Applying fix..."
    ssh "$REMOTE" "cp /usr/share/glvnd/egl_vendor.d/10_nvidia.json ~/miniconda3/envs/$REMOTE_CONDA_ENV/share/glvnd/egl_vendor.d/"
    echo "  EGL fix applied."
fi
echo ""

# Step 4: Check GPU 0-3 availability
echo "[Step 4/5] Checking GPU availability..."
ssh "$REMOTE" "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -i 0,1,2,3"
echo ""
echo "  If GPUs look busy, Ctrl+C now. Continuing in 5s..."
sleep 5
echo ""

# Step 5: Launch 4-GPU training
echo "[Step 5/5] Launching 4-GPU training (run_name=$RUN_NAME)..."
ssh "$REMOTE" bash -s <<REMOTE_LAUNCH
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate $REMOTE_CONDA_ENV
  cd $REMOTE_PROJECT

  # Kill any existing training on these GPUs
  pkill -f "train_multigpu.py.*$RUN_NAME" 2>/dev/null || true
  sleep 1

  nohup torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    scripts/rsl_rl/train_multigpu.py \
    --task=Magiclab-Z1-12dof-Velocity \
    --run_name=$RUN_NAME \
    --headless \
    --distributed \
    --num_envs=$NUM_ENVS \
    --max_iterations=$MAX_ITER \
    > $LOG_FILE 2>&1 &

  echo "  Training PID: \$!"
  sleep 2
  echo ""
  echo "  Verify: tail -n 20 $LOG_FILE"
REMOTE_LAUNCH
echo ""

echo "=========================================="
echo " Deployment complete!"
echo " Run name:  $RUN_NAME"
echo " Log file:  $LOG_FILE on $REMOTE"
echo " Check:     ssh $REMOTE 'tail -30 $LOG_FILE'"
echo "=========================================="
