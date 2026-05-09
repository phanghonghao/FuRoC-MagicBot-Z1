# FuRoC-MagicBot-Z1

RL locomotion training pipeline for **MagicBot Z1 12DOF bipedal robot**, built on Isaac Lab + rsl_rl.

## Results

### Pipeline Demo

<p align="center">
  <img src="docs/github_readme/pipeline_p1p2_demo.gif" alt="P1-P2 Pipeline Demo (Isaac Lab + MuJoCo)" width="60%">
</p>

> **4-phase curriculum learning** — P1 Coarse → P1 Fine → P2 Coarse → P2 Fine. Left column: Isaac Lab simulation. Right column: MuJoCo sim2sim validation.

### Curriculum Reward Trends

<p align="center">
  <img src="docs/github_readme/curriculum_reward_trends.png" alt="Curriculum Reward Trends" width="90%">
</p>

> Reward curves across 4 sub-phases. P1 (flat terrain, bootstrap → standing), P2 (flat, velocity tracking). Each phase resumes from the best checkpoint of the previous phase.

### P2 Fine Reward Decomposition

<p align="center">
  <img src="docs/github_readme/reward_trend_p2_fine.png" alt="P2 Fine Reward Trend" width="45%">
  &nbsp;
  <img src="docs/github_readme/reward_decomposition_p2_fine.png" alt="P2 Fine Reward Decomposition" width="45%">
</p>

> **Left**: P2 Fine total reward trend. **Right**: Individual reward component decomposition — velocity tracking, orientation, base height, foot contact, action rate penalty, and torque penalty.

### Sim2Sim Gap: Terrain-Trained Policies in MuJoCo

<p align="center">
  <img src="docs/github_readme/p3_fine_sim2sim_broken.gif" alt="P3 Fine sim2sim broken in MuJoCo" width="45%">
</p>

> **P3 Fine policy (gentle terrain training) deployed to MuJoCo** — the robot repeatedly falls. Flat-trained policies (P1, P2) transfer successfully, but terrain-trained policies fail due to sim2sim physics gap (contact/friction model differences). One root cause: policies trained on rough terrain cannot walk on flat ground in a different simulator.

### Pre-trained Models

| Phase | Policy | Path | Description |
|-------|--------|------|-------------|
| P1 Coarse | Standing | `models/p/p1_coarse/p1_coarse_policy.pt` | Bootstraps standing from random init |
| P1 Fine | Standing | `models/p/p1_fine/p1_fine_policy.pt` | Fine-tuned stable standing on flat terrain |
| P2 Coarse | Locomotion | `models/p/p2_coarse/p2_coarse_policy.pt` | Initial velocity tracking on flat terrain |
| P2 Fine | Locomotion | `models/p/p2_fine/p2_fine_policy.pt` | Fine-tuned velocity tracking with gait shaping |

## Directory Structure

```
Magicbot_Z1/
├── magiclab_rl_lab/          # RL framework (fork, z1-custom branch)
├── magicbot-z1_description/  # URDF/Mesh (official)
├── magicbot-z1_sdk/          # Robot SDK (official)
├── configs/                  # Custom env configs & scripts
├── docs/                     # Training plans, analysis, plots, demo GIFs
│   ├── github_readme/        # Demo GIFs & plots for README
│   ├── pipeline_p1p2_demo.gif
│   ├── curriculum_reward_trends.png
│   ├── reward_decomposition_p2_fine.png
│   └── reward_trend_p2_fine.png
├── models/
│   └── p/                    # Pipeline policy checkpoints (Git LFS)
│       ├── p1_coarse/
│       ├── p1_fine/
│       ├── p2_coarse/
│       └── p2_fine/
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

## 5-Phase Automated Pipeline

Fully automated training pipeline with overfitting detection, auto-rollback, and phase advancement.

| Phase | Terrain | Key Goal | Sub-phases | Status |
|-------|---------|----------|------------|--------|
| P1 | Flat | Bootstrap standing | coarse → fine | Done |
| P2 | Flat | Velocity tracking | coarse → fine | Done |
| P3 | 70% flat + 30% gentle grid | Light terrain walking | coarse → fine | Done |
| P3b | 50% flat + 30% grid + 10% stairs + 10% boxes | Intermediate terrain | coarse → fine | In progress |
| P4 | Flat + grid + stairs + gap + boxes | Rough terrain | coarse → fine | Planned |
| P5 | Full terrain + rails | Complex + high speed | coarse → fine | Planned |

Each sub-phase: config generation → distributed PPO training → overfitting detection → video recording → advance. Orchestrator auto-detects 5 failure signals (reward decline, policy collapse, action explosion, entropy collapse, value divergence) and rolls back if needed.

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
