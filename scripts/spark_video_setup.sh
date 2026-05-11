#!/bin/bash
# MagicBot Z1 Video Recording on Spark (DGX Spark, aarch64)
# Run this script on your LOCAL machine (Windows/Git Bash)
# Prerequisites: SSH access to both RTX6000 and Spark

set -e

SPARK="spark"
SPARK_USER="zentek"
RTX="phh@192.168.120.155"
CKPT_DIR="2026-05-01_04-44-07_s1_flat_retry"
CKPT="model_30500.pt"

echo "=== Step 1: Transfer files from RTX6000 to Spark ==="
echo "Transferring source code + assets (~19MB)..."
scp -r ${RTX}:~/magiclab_rl_lab/source/magiclab_rl_lab/ /tmp/magiclab_rl_lab_transfer/
scp -r ${RTX}:~/magiclab_rl_lab/scripts/ /tmp/magiclab_rl_lab_transfer/

echo "Transferring checkpoint (~7MB)..."
mkdir -p /tmp/magiclab_rl_lab_transfer/logs/rsl_rl/magiclab_z1_12dof_velocity/${CKPT_DIR}/
scp ${RTX}:~/magiclab_rl_lab/logs/rsl_rl/magiclab_z1_12dof_velocity/${CKPT_DIR}/${CKPT} /tmp/magiclab_rl_lab_transfer/logs/rsl_rl/magiclab_z1_12dof_velocity/${CKPT_DIR}/

echo "Uploading to Spark..."
ssh ${SPARK} "mkdir -p ~/magiclab_rl_lab/source/ ~/magiclab_rl_lab/scripts/ ~/magiclab_rl_lab/logs/rsl_rl/magiclab_z1_12dof_velocity/${CKPT_DIR}/"
scp -r /tmp/magiclab_rl_lab_transfer/source/magiclab_rl_lab/ ${SPARK}:~/magiclab_rl_lab/source/
scp -r /tmp/magiclab_rl_lab_transfer/scripts/ ${SPARK}:~/magiclab_rl_lab/
scp /tmp/magiclab_rl_lab_transfer/logs/rsl_rl/magiclab_z1_12dof_velocity/${CKPT_DIR}/${CKPT} ${SPARK}:~/magiclab_rl_lab/logs/rsl_rl/magiclab_z1_12dof_velocity/${CKPT_DIR}/

echo "=== Step 2: Fix paths and install on Spark ==="
ssh ${SPARK} bash -s << 'SPARK_SETUP'
cd ~/magiclab_rl_lab

# Fix MAGICLAB_ROS_DIR for Spark
sed -i 's|MAGICLAB_ROS_DIR = .*|MAGICLAB_ROS_DIR = "/home/zentek/magiclab_rl_lab"|' \
    source/magiclab_rl_lab/magiclab_rl_lab/assets/robots/magiclab.py

# Install the package
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
pip install -e source/magiclab_rl_lab

echo "Setup complete!"
SPARK_SETUP

echo "=== Step 3: Record video on Spark ==="
ssh ${SPARK} bash -s << 'SPARK_RECORD'
export DISPLAY=:1
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
cd ~/magiclab_rl_lab

python scripts/rsl_rl/play_z1_video.py \
    --task Magiclab-Z1-12dof-Velocity \
    --checkpoint logs/rsl_rl/magiclab_z1_12dof_velocity/2026-05-01_04-44-07_s1_flat_retry/model_30500.pt \
    --headless --video --video_length 200 --num_envs 1 --device=cuda:0

echo "Recording done!"
SPARK_RECORD

echo "=== Step 4: Download video ==="
CKPT_PATH="logs/rsl_rl/magiclab_z1_12dof_velocity/2026-05-01_04-44-07_s1_flat_retry"
scp ${SPARK}:~/magiclab_rl_lab/${CKPT_PATH}/videos/play/rl-video-step-0.mp4 ./z1_s1_flat_retry_m30500_isaaclab.mp4

echo "Done! Video saved to ./z1_s1_flat_retry_m30500_isaaclab.mp4"
