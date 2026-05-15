# FuRoC-MagicBot-Z1

RL locomotion training pipeline for **MagicBot Z1 12DOF bipedal robot**, built on Isaac Lab + rsl_rl.

## Pipeline Overview

<p align="center">
  <img src="docs/github_readme/pipeline_flow.svg" alt="5-Phase Curriculum Learning Pipeline" width="95%">
</p>

> **5-phase curriculum learning** with terrain progression: Flat → Gentle → Rough → Full. Each phase resumes from the best checkpoint of the previous phase. Currently retraining from P2 Fine with joint_mirror symmetry reward to fix left-right asymmetry before advancing to terrain phases.

## Demo

### P1–P2 Pipeline Demo

<p align="center">
  <img src="docs/github_readme/pipeline_p1p2_demo.gif" alt="P1-P2 Pipeline Demo (Isaac Lab + MuJoCo)" width="60%">
</p>

> P1 Coarse → P1 Fine → P2 Coarse → P2 Fine. Left column: Isaac Lab simulation. Right column: MuJoCo sim2sim validation.

## Results

### Curriculum Reward Trends

<p align="center">
  <img src="docs/github_readme/curriculum_reward_trends.png" alt="Curriculum Reward Trends" width="90%">
</p>

> Reward curves across sub-phases. P1 (flat terrain, bootstrap → standing), P2 (flat, velocity tracking). Each phase resumes from the best checkpoint of the previous phase.

### Left-Right Joint Asymmetry

<p align="center">
  <img src="docs/github_readme/joint_asymmetry_p2_vs_p3.png" alt="Left vs Right Joint Angles P2 vs P3" width="95%">
</p>

> Time-series of left (blue) vs right (red) joint angles. **Top row (P2 Fine, flat terrain)**: joints are roughly symmetric (offset < 0.03 rad). **Bottom row (P3 Coarse, gentle terrain)**: significant offset appears — hip pitch (−0.37 rad), hip yaw (−0.52 rad), knee pitch (+0.39 rad).

<p align="center">
  <img src="docs/github_readme/joint_asymmetry_barplot.png" alt="Joint Asymmetry Bar Plot" width="85%">
</p>

> Quantitative comparison of left-right asymmetry across phases. P3 Coarse shows 10–20x larger mean offset than P2 Fine.
>
> **Root cause**: The reward function only penalizes each joint's deviation from its default position (`joint_deviation_l1`), but never enforces left-right correspondence. On flat terrain (P2) the optimal gait happens to be symmetric, but random terrain (P3) exposes this gap — PPO freely converges to an asymmetric local optimum where left and right legs use fundamentally different joint angles, yet still scores high reward.
>
> **Fix**: Add a symmetry reward term `|qpos_left - qpos_right|` or enable Isaac Lab's built-in `RslRlSymmetryCfg(use_mirror_loss=True)`. Can resume from current P3b Fine checkpoint without retraining from scratch.

## Pre-trained Models

| Phase | Policy | Path | Description |
|-------|--------|------|-------------|
| P1 Coarse | Standing | `models/p/p1_coarse/p1_coarse_policy.pt` | Bootstraps standing from random init |
| P1 Fine | Standing | `models/p/p1_fine/p1_fine_policy.pt` | Fine-tuned stable standing on flat terrain |
| P2 Coarse | Locomotion | `models/p/p2_coarse/p2_coarse_policy.pt` | Initial velocity tracking on flat terrain |
| P2 Fine | *Retraining* | — | Retraining with joint_mirror symmetry reward |

## 5-Phase Automated Pipeline

Fully automated training pipeline with overfitting detection, auto-rollback, and phase advancement.

| Phase | Terrain | Key Goal | Sub-phases | Status |
|-------|---------|----------|------------|--------|
| P1 | Flat | Bootstrap standing | coarse → fine | Done ✅ |
| P2 | Flat | Velocity tracking | coarse → fine | Retraining 🔄 (joint_mirror) |
| P3 | 70% flat + 30% gentle grid | Light terrain walking | coarse → fine | Planned ⏳ |
| P4 | Flat + grid + stairs + gap + boxes | Rough terrain | coarse → fine | Planned ⏳ |
| P5 | Full terrain + rails | Complex + high speed | coarse → fine | Planned ⏳ |

Each sub-phase: config generation → distributed PPO training → overfitting detection → video recording → advance. Orchestrator auto-detects 5 failure signals (reward decline, policy collapse, action explosion, entropy collapse, value divergence) and rolls back if needed.

## Directory Structure

```
Magicbot_Z1/
├── magiclab_rl_lab/          # RL framework (fork, z1-custom branch)
├── magicbot-z1_description/  # URDF/Mesh (official)
├── magicbot-z1_sdk/          # Robot SDK (official)
├── configs/                  # Custom env configs & scripts
├── docs/
│   └── github_readme/        # Demo GIFs, plots & SVG for README
├── models/
│   └── p/                    # Pipeline policy checkpoints (Git LFS)
│       ├── p1_coarse/  p1_fine/
│       ├── p2_coarse/  p2_fine/
├── videos/                   # Training demo videos (Git LFS)
├── IsaacLab/                 # Isaac Lab framework (.gitignored)
└── README.md
```

## Submodules

| Submodule | Source | Branch |
|-----------|--------|--------|
| `magiclab_rl_lab` | [phanghonghao/magiclab_rl_lab](https://github.com/phanghonghao/magiclab_rl_lab) (fork) | `z1-custom` |
| `magicbot-z1_description` | [MagiclabRobotics/magicbot-z1_description](https://github.com/MagiclabRobotics/magicbot-z1_description) | main |
| `magicbot-z1_sdk` | [MagiclabRobotics/magicbot-z1_sdk](https://github.com/MagiclabRobotics/magicbot-z1_sdk) | main |

## Quick Start

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/phanghonghao/FuRoC-MagicBot-Z1.git
cd FuRoC-MagicBot-Z1
```

### 2. Install Isaac Lab

```bash
# Isaac Lab must be installed separately (excluded from repo)
# See: https://isaac-sim.github.io/IsaacLab/
# Symlink into Magicbot_Z1/IsaacLab/
```

### 3. Train

```bash
cd magiclab_rl_lab
bash train_bash.sh
```

### 4. Evaluate / Record Video

```bash
# Play trained policy
python scripts/rsl_rl/play_z1_video.py --task=<version>

# Sim2sim (MuJoCo)
python sim2sim/mujoco_deploy.py --ckpt=<path_to_model>

# Deploy to robot
python deploy/robot_deploy.py
```

## Documentation

- [Training Plan](docs/Z1_Locomotion_Training_Plan.md)
- [Training Analysis](docs/Z1_Training_Analysis.md)
- [TODO & Naming Convention](docs/TODO.md)
- [Framework Guide](docs/FRAMEWORK.md)

## Hardware

- **GPU**: 4 × RTX 6000D (85 GB VRAM each)
- **Training**: `torchrun` distributed PPO, 16,384 parallel envs (4,096/GPU)
- **Throughput**: ~330K steps/s (4 GPUs)
- **Framework**: Isaac Lab + rsl_rl
