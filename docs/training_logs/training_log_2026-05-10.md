# Training Log — 2026-05-10

---

## [01:19] Session Summary — p3b_stable launch attempt + MuJoCo terrain development

### Completed

| # | Item | Details |
|---|------|---------|
| 1 | p3b_stable training launch | Attempted to launch on RTX with anti-overfitting PPO (entropy=0.03, lr=3e-5, kl=0.005). Failed: GPU 0-3 occupied by Justin's VLLM TP0-TP3 (~75GB each). PhysX OOM on GPU 0. |
| 2 | GPU resource survey | Identified: GPU 0-3 busy (Justin VLLM), GPU 4-7 idle. torchrun assigns from GPU 0 — cannot skip to GPU 4-7. |
| 3 | MuJoCo terrain support | Added `--terrain p3b` to `mujoco_manual.py`. Generates heightfield matching p3b Isaac Sim config (flat 50%, random_grid 30%, stairs 10%, boxes 10%). Uses MuJoCo Python API to inject hfield data after XML load (bypasses binary file format issues). |
| 4 | MuJoCo hfield format debugging | Discovered MuJoCo 3.8.0 hfield file format is incompatible with standard binary dumps. Solution: define hfield in XML without `file` attribute, set `model.hfield_data[:]` via Python API. |
| 5 | s1_flat sim2sim verification | Confirmed `mujoco_manual.py` works correctly with s1_flat model: 20s walk, 3.6m distance, 0 falls. |
| 6 | Bug fix: p3b_fine MuJoCo sim2sim failure | **Root cause**: p3b_fine model fails in MuJoCo (9 falls/10s on flat ground, 0 falls but 0 movement on terrain). s1_flat works fine. Suspected sim2sim gap from terrain curriculum training — policy may depend on Isaac Sim-specific physics or observation features not replicated in MuJoCo. **Status**: Unresolved. |

### Uncompleted / Blocked

| # | Item | Blocker | Next Step |
|---|------|---------|-----------|
| 1 | p3b_stable training launch | GPU 0-3 occupied by Justin's VLLM. torchrun can't use GPU 4-7. | Wait for Justin to finish, or use single-GPU with `--device cuda:4` (requires code change for single GPU launch) |
| 2 | p3b_fine MuJoCo video with terrain | sim2sim broken for p3b_fine model (works for s1_flat) | Debug observation mismatch, or use Isaac Sim recording on single GPU (cuda:4) |
| 3 | p3b_stable training | Not yet started | Re-launch after GPU 0-3 freed, or modify launch for single GPU |

### Key Decisions

- **Abandoned p4/p5 rough terrain**: Decided in previous session to stop p4_fine and focus on indoor/flat walking (p3b terrain level sufficient)
- **Anti-overfitting PPO params**: entropy_coef 0.03, lr 3e-5, desired_kl 0.005 — chosen to address consistent overfitting across all phases
- **Terrain in MuJoCo**: Chose Python API hfield injection over file-based approach due to MuJoCo 3.8.0 binary format incompatibility
- **p3b_fine sim2sim not viable**: Robot can't walk in MuJoCo — likely needs Isaac Sim recording instead

### Files Modified

| File | Change |
|------|--------|
| `~/magiclab_rl_lab/sim2sim/mujoco_manual.py` (RTX) | Added `--terrain` arg, `generate_terrain_data()`, `load_model_with_terrain()`. Terrain hfield injected via Python API. |
| `~/magiclab_rl_lab/tmp/phase_configs/p3b_stable/ppo_override_cfg.py` (RTX) | Created in previous session: anti-overfitting PPO config |
| `~/magiclab_rl_lab/source/.../velocity_env_cfg.py` (RTX) | Restored to p3b terrain config (from p3b_fine backup) |

---

## [02:10] Session Summary — MuJoCo terrain video recording pipeline (p1-p3b)

### Completed

| # | Item | Details |
|---|------|---------|
| 1 | p1_fine MuJoCo video (flat) | Recorded + downloaded + labeled. 1000 steps, 0 falls, 0.2m movement. File: `videos/p/p1_fine/p1_fine_m2800_sim2sim_mujoco.mp4` (744K) |
| 2 | p2_fine MuJoCo video (flat) | Recorded + downloaded + labeled. 1000 steps, 0 falls, 4.0m movement. File: `videos/p/p2_fine/p2_fine_m3000_sim2sim_mujoco.mp4` (381K) |
| 3 | p3_fine MuJoCo video (gentle terrain) | Recorded + downloaded + labeled. 1000 steps, 20 falls (sim2sim broken). File: `videos/p/p3_fine/p3_fine_m10700_sim2sim_mujoco.mp4` (795K) |
| 4 | p3b_fine MuJoCo video (intermediate terrain) | Recorded + downloaded + labeled. 1000 steps, 0 falls but 0m movement (sim2sim broken). File: `videos/p/p3b_fine/p3b_fine_m14600_sim2sim_mujoco.mp4` (632K) |
| 5 | Training params download | Downloaded `params/` (agent.yaml, env.yaml, deploy.yaml) for all 4 phases: p1_fine, p2_fine, p3_fine, p3b_fine |
| 6 | Video labeling | All 4 videos labeled with `label_video.py` (run name, model, terrain type, sim2sim status). Labels replaced originals. |

### Video Summary

| Phase | Terrain | Model | Sim2Sim Result | Local Path |
|-------|---------|-------|---------------|------------|
| p1_fine | flat | model_2800 | OK (0 falls, 0.2m) | `videos/p/p1_fine/p1_fine_m2800_sim2sim_mujoco.mp4` |
| p2_fine | flat | model_3000 | OK (0 falls, 4.0m) | `videos/p/p2_fine/p2_fine_m3000_sim2sim_mujoco.mp4` |
| p3_fine | p3 gentle | model_10700 | BROKEN (20 falls) | `videos/p/p3_fine/p3_fine_m10700_sim2sim_mujoco.mp4` |
| p3b_fine | p3b intermediate | model_14600 | BROKEN (0m movement) | `videos/p/p3b_fine/p3b_fine_m14600_sim2sim_mujoco.mp4` |

### Uncompleted / Blocked

| # | Item | Blocker | Next Step |
|---|------|---------|-----------|
| 1 | p3b_stable training launch | GPU 0-3 occupied by Justin's VLLM (~75GB each). torchrun assigns from GPU 0. | Wait for Justin to finish, or use `--device cuda:4` single GPU launch |
| 2 | p3/p3b sim2sim gap | Terrain-trained models (p3, p3b) fail in MuJoCo — fall or freeze. Flat models (p1, p2) work fine. | Debug observation mismatch (contact forces, friction model differences), or record via Isaac Sim on cuda:4 |
| 3 | Isaac Lab videos for p1-p3b | Not attempted — requires training to be stopped (Omniverse Kit grabs all 8 GPUs) | Record when training is paused/stopped |

### Key Findings

- **sim2sim gap confirmed**: Flat-trained policies (p1, p2) transfer to MuJoCo successfully. Terrain-trained policies (p3, p3b) do NOT transfer — they either fall repeatedly (p3: 20 falls) or freeze (p3b: 0m movement, 0 falls).
- **Root cause hypothesis**: Terrain curriculum introduces observation dependencies on Isaac Sim-specific contact/friction physics that MuJoCo does not replicate identically. The policy learns to exploit Isaac Sim terrain features that don't exist in MuJoCo.
- **Terrain generation working**: MuJoCo hfield injection via Python API works correctly for all terrain types (flat, p3 gentle, p3b intermediate).

---

## [Session] 本地 MuJoCo 键盘测试 + 相位感知部署

### Completed

| # | Item | Details |
|---|------|---------|
| 1 | mujoco_manual.py 重构 | 合并 terrain patch → 正式版 `sim2sim/mujoco_manual.py`。添加 `--phase` 参数 + `PHASE_TERRAIN` 映射（p1/p2→flat, p3→p3, p3b→p3b, p4→p3b）。`--terrain` 优先于 `--phase`。 |
| 2 | yaml.safe_load 修复 | `yaml.safe_load` → `yaml.unsafe_load`，修复 deploy.yaml 中 `!!python/object/apply:builtins.slice` 解析失败。 |
| 3 | MUJOCO_GL 平台兼容 | Windows 不设 `MUJOCO_GL=egl`（仅 Linux 需要），修复 Windows 启动报错。 |
| 4 | PD gains 修复 | **Root cause**: 训练已用 IdealPDActuator（显式 PD），跟 MuJoCo 一样，不需要 30% KD boost。之前硬编码的 DEFAULT_KD [5.2, 5.2, 6.5, 3.9] 是旧 ImplicitActuator 时代的补偿。**Fix**: KP/KD 从 deploy.yaml 加载（[4.0, 4.0, 5.0, 3.0]），default_joint_pos 硬编码（deploy.yaml 导出 joint 顺序与 MuJoCo 不匹配）。 |
| 5 | 本地 p2_fine 键盘测试 | **结果**: 150+ 秒仅 1 次摔倒，z≈0.66 稳定站立。对比修复前 256 次/200 秒，大幅改善。 |
| 6 | Viewer 问题 | MuJoCo viewer 未能弹出（headless 模式运行）。已加错误打印，待下次调试。 |

### 测试结果

| 测试 | Phase | 地形 | 结果 | 备注 |
|------|-------|------|------|------|
| p2_fine 修复前 | p2 | flat | 256 falls / 200s | KD=5.2 (30% boost)，deploy.yaml 覆盖后反而更差 |
| p2_fine 修复后 | p2 | flat | 1 fall / 150s | KD=4.0 (训练原值)，稳定站立 |

### Bug Fixes

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| YAML 解析失败 | `yaml.safe_load` 不支持 `!!python/object/apply:builtins.slice` | → `yaml.unsafe_load` |
| Windows MUJOCO_GL 报错 | EGL 仅 Linux 支持，Windows 应使用默认 wgl | 添加 `os.name != "nt"` 判断 |
| 机器人反复摔倒 | DEFAULT_KD 硬编码 30% boost (5.2)，但训练已用 IdealPD (显式 PD)，boost 反而有害 | KP/KD 从 deploy.yaml 加载，不 boost |
| MuJoCo viewer 不弹出 | `launch_passive()` 异常被静默吞掉 | 加 `as e` 打印具体错误 |

### Files Modified

| File | Change |
|------|--------|
| `mujoco_terrain_patch.py` → `sim2sim/mujoco_manual.py` | 合并重构：--phase 参数、yaml 修复、MUJOCO_GL 修复、PD gains 修复 |
| `sim2sim/mujoco_manual.py` (本地) | 新建 |
| `magiclab_rl_lab/sim2sim/mujoco_manual.py` (RTX 镜像) | 同步更新 |

### Next Steps

- [ ] 调试 MuJoCo viewer 不弹出问题（打印具体异常）
- [ ] 测试 --phase p3b + 地形键盘控制
- [ ] 上传到 RTX: `scp sim2sim/mujoco_manual.py phh@192.168.120.155:~/magiclab_rl_lab/sim2sim/`
