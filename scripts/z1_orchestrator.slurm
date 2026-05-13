#!/bin/bash
#SBATCH --job-name=z1_5phase
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=/home/phh/magiclab_rl_lab/logs/slurm/%j.out
#SBATCH --error=/home/phh/magiclab_rl_lab/logs/slurm/%j.err
#SBATCH --chdir=/home/phh/magiclab_rl_lab

# Force GPU 0-3 (avoid GPU 6 which has bypass process)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab

# Resume 5-phase orchestrator from saved state (p3_fine)
python -u scripts/automation/phase_orchestrator.py \
    --plan training_plans/z1_5phase_plan.yaml \
    --num-gpus 4 \
    --poll-interval 120
