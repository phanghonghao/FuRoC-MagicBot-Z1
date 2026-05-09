# 配置生成

Orchestrator 在启动训练前会做两件事：
1. 生成 `velocity_env_cfg.py`（地形 + reward 权重）→ 覆盖到 `source/.../velocity_env_cfg.py`
2. 生成 `ppo_override_cfg.py`（PPO 超参）→ 通过 `--agent_cfg` 传给训练脚本

## 用 Orchestrator Dry-Run 生成配置

```bash
cd ~/magiclab_rl_lab
source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab

# 只生成配置，不启动训练
python -u scripts/automation/phase_orchestrator.py \
    --plan training_plans/z1_5phase_plan.yaml \
    --start-from p3b_coarse \
    --fresh \
    --dry-run
```

输出位置：`tmp/phase_configs/<子阶段>/`

## 各阶段对应的环境配置

需要手动修改时，编辑：
```
source/magiclab_rl_lab/magiclab_rl_lab/tasks/locomotion/robots/z1/12dof/velocity_env_cfg.py
```

关键字段：
- `COBBLESTONE_ROAD_CFG` — 地形生成器配置
- `rewards` dict — 各 reward 权重
- `commands` — 速度指令范围

## PPO 参数覆盖

PPO 参数通过 `--agent_cfg` 传入，不影响原始配置文件。只需指定需要覆盖的参数。

手动创建 PPO override 文件：

```python
"""PPO override for manual training."""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class PhasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 100
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.015,          # ← 按阶段调整
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,        # ← 按阶段调整
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

保存为 `tmp/ppo_manual.py`，然后 `--agent_cfg=tmp/ppo_manual.py`。
