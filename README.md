# FuRoC-MagicBot-Z1

RL locomotion training pipeline for **MagicBot Z1 12DOF bipedal robot**, built on Isaac Lab + rsl_rl.

## Directory Structure

```
Magicbot_Z1/
├── magiclab_rl_lab/          # RL framework (fork, z1-custom branch)
├── magicbot-z1_description/  # URDF/Mesh (official)
├── magicbot-z1_sdk/          # Robot SDK (official)
├── configs/                  # Custom env configs & scripts
├── docs/                     # Training plans, analysis, TODOs
├── models/                   # Best policy checkpoints (Git LFS)
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

## Training Pipeline

| Stage | Terrain | Goal | Status |
|-------|---------|------|--------|
| s1 | Flat | Standing | Done |
| s2 | Flat | Walking | Done |
| s3 | 50% flat + 50% gentle grid | Light terrain | Done |
| s4 | Flat + grid + stairs + gap + boxes | Rough terrain | In progress |
| s5 | Full terrain + rails | Complex + high speed | Planned |

## Documentation

- [Training Plan](docs/Z1_Locomotion_Training_Plan.md)
- [Training Analysis](docs/Z1_Training_Analysis.md)
- [TODO & Naming Convention](docs/TODO.md)
- [Framework Guide](docs/FRAMEWORK.md)

## Hardware

- **GPU**: RTX 6000D (85 GB VRAM)
- **Environments**: 4096 parallel
- **Training time**: ~28-35h / 50K iterations
