# RTX 训练 → 策略导出 → 视频录制 完整流程

> 本文档记录已验证通过的端到端流程：RTX6000 上训练 → JIT 导出 → RTX Isaac Sim / MuJoCo 录视频。
>
> 最后验证：2026-05-07，FPS 修复（50fps = 控制频率），video_length=500（10s 实时）。

---

## 1. 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        RTX 6000D (x86_64)                       │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────────┐ │
│  │ 训练      │───→│ export_jit.py│───→│ exported/policy.pt     │ │
│  │ train.py  │    │ (纯 PyTorch) │    │ (JIT, ~1.1MB)         │ │
│  └──────────┘    └──────────────┘    └───────┬────────────────┘ │
│                                              │                  │
│  ┌──────────────────┐  ┌─────────────────┐   │                  │
│  │ Isaac Sim 录制    │  │ MuJoCo 录制      │◄──┤ (也支持原始      │
│  │ play_z1_video.py │  │ mujoco_manual.py │   │   checkpoint)    │
│  │ --headless       │  │ --record --headless│  │                  │
│  │ fps=50 (关键!)   │  │ fps=50 (自动)    │   │                  │
│  └──────────────────┘  └─────────────────┘   │                  │
└──────────────────────────────────────────────┼──────────────────┘
                                               │ scp (raw .mp4)
                    ┌───────────────────────────┤
                    │ 本地 Windows               │
                    │ 1. ffmpeg re-encode        │
                    │ 2. label_video.py (1次!)   │
                    └───────────────────────────┘
```

<!-- Spark (DGX) 已归还。旧流程参考:
  - spark_play.py: Isaac Sim 录制脚本（已废弃，camera tracking 逻辑已移植到 play_z1_video.py）
  - spark_deploy.sh: Spark 一键部署脚本（已废弃）
  - 旧录制使用 --checkpoint 模式，RTX 上统一用 --policy (JIT) 模式
-->

### 关键参数：FPS 必须匹配控制频率

| 参数 | 值 | 说明 |
|------|-----|------|
| sim.dt | 0.002s | 物理仿真频率 500Hz |
| decimation | 10 | 控制频率 = 500/10 = **50Hz** |
| 每步模拟时间 | 0.02s | = dt × decimation |
| **RecordVideo fps** | **50** | **必须 = 控制频率，否则视频播放速度错误** |
| video_length | 500 steps | 500 × 0.02s = **10s 实时视频** |

> **踩坑记录**：`gym.wrappers.RecordVideo` 默认 fps=30（当 env 未设 `metadata["render_fps"]` 时）。
> Isaac Lab 环境未设此字段，导致 30fps ≠ 50Hz 控制频率，视频播放速度变成 0.6x 慢放。
> 已在 `play_z1_video.py` 中显式传 `fps=50` 修复。

### 版本兼容性矩阵

| 组件 | RTX 6000 | 说明 |
|------|----------|------|
| rsl-rl | 3.0.1 (`model_state_dict`) | `play_z1_video.py --checkpoint` 直接加载 |
| IsaacLab | 0.47.2 | 源码同步 |
| IsaacSim | 4.5.0 | `--headless` 直接录制（无需 Xvfb） |
| PyTorch | 2.11.0+cu128 | JIT 模型导出 + 加载 |
| Python | 3.10 | |
| MuJoCo | EGL offscreen | 无需 X Server，`MUJOCO_GL=egl` |
| empirical_normalization | **False** | JIT 导出无需包含归一化层，两种模式行为一致 |

### 核心设计：为什么用 JIT 导出

rsl-rl 3.x 和 5.x 的 checkpoint 格式不兼容。通过 JIT 导出 actor 网络，得到一个纯 `torch.jit.script` 模型文件，**任何平台、任何 rsl-rl 版本都能用 `torch.jit.load` 加载**。

> 验证：本训练 `empirical_normalization = False`，checkpoint 中无归一化统计量。
> JIT 导出的 actor 与 OnPolicyRunner 的 `act_inference` 产生完全相同的输出。

---

## 2. 文件清单

| 文件 | 位置 | 用途 |
|------|------|------|
| `export_jit.py` | RTX: `~/magiclab_rl_lab/scripts/export_jit.py` | 从 checkpoint 导出 JIT 模型 |
| `play_z1_video.py` | RTX: `~/magiclab_rl_lab/scripts/rsl_rl/play_z1_video.py` | Isaac Sim 录制（`--checkpoint` 或 `--policy`，fps=50） |
| `mujoco_manual.py` | RTX: `~/magiclab_rl_lab/sim2sim/mujoco_manual.py` | MuJoCo sim2sim 录制（EGL offscreen，fps=50） |
| `rtx_record_video.sh` | 本地: `D:\Desktop_Files\GPU-Train\RTX6000\rtx_record_video.sh` | 一键录制 Isaac Sim + MuJoCo |
| `label_video.py` | 本地: `Magicbot_Z1/scripts/label_video.py` | 视频标签叠加（保留输入 fps） |

---

## 3. 完整录制流程（6 步）

### 3.1 RTX 上导出 JIT（如果尚未导出）

```bash
ssh phh@192.168.120.155
cd ~/magiclab_rl_lab && source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab

# 导出最佳 checkpoint
python scripts/export_jit.py \
    --checkpoint logs/rsl_rl/<RUN>/model_<N>.pt
# 输出: logs/rsl_rl/<RUN>/exported/policy.pt
```

### 3.2 RTX 上 Isaac Sim 录制（headless, fps=50）

```bash
ssh phh@192.168.120.155 "cd ~/magiclab_rl_lab && \
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab && \
  python -u scripts/rsl_rl/play_z1_video.py \
    --policy logs/rsl_rl/<RUN>/exported/policy.pt \
    --headless --video --video_length 500 --num_envs 1 --device=cuda:0"
# 输出: logs/rsl_rl/<RUN>/videos/play/rl-video-step-0.mp4 (500帧, 50fps, 10s)
```

**play_z1_video.py 参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--checkpoint` | 原始 checkpoint 路径（OnPolicyRunner） | — |
| `--policy` | JIT policy 路径 | — |
| `--num_envs` | 并行环境数 | 1 |
| `--video` | 录制视频 | False |
| `--video_length` | 视频步数（500 = 10s） | **500** |
| `--max_steps` | 非录制模式最大步数 | 800 |
| `--no_camera_track` | 禁用摄像头追踪 | False |
| `--camera_distance` | 摄像头距离 | 3.5 |
| `--camera_height` | 摄像头高度 | 1.5 |

### 3.3 RTX 上 MuJoCo 录制（EGL, fps=50）

```bash
ssh phh@192.168.120.155 "cd ~/magiclab_rl_lab && \
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab && \
  python -u sim2sim/mujoco_manual.py \
    --mjcf ~/magicbot-z1_description/mjcf/MAGICBOTZ1.xml \
    --checkpoint logs/rsl_rl/<RUN>/exported/policy.pt \
    --record /tmp/<save_name>.mp4 \
    --duration 10 --vel_x 0.5 --headless"
# 输出: /tmp/<save_name>.mp4 (fps=50, 10s)
```

### 3.4 下载到本地

```bash
VIDDIR="D:/Desktop_Files/GPU-Train/RTX6000/Magicbot_Z1/videos/<stage>"
mkdir -p "$VIDDIR"

# Isaac Sim
scp phh@192.168.120.155:~/magiclab_rl_lab/logs/rsl_rl/<RUN>/videos/play/rl-video-step-0.mp4 \
    "$VIDDIR/<stage>_raw_isaaclab.mp4"

# MuJoCo
scp phh@192.168.120.155:/tmp/<save_name>.mp4 \
    "$VIDDIR/<stage>_raw_mujoco.mp4"
```

### 3.5 本地 ffmpeg re-encode（Isaac Sim 视频需要）

Isaac Sim 输出的 mp4 容器格式可能无法被 PyAV 直接读取，需要 ffmpeg 重编码：

```bash
python -c "
import imageio_ffmpeg, subprocess, glob
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
for raw in glob.glob('D:/Desktop_Files/GPU-Train/RTX6000/Magicbot_Z1/videos/*/*_raw_isaaclab.mp4'):
    out = raw.replace('_raw_isaaclab.mp4', '_isaaclab_tmp.mp4')
    subprocess.run([ffmpeg, '-y', '-i', raw, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out])
    print(f'Re-encoded: {out}')
"
```

> MuJoCo 视频由 `imageio.mimwrite` 生成，格式标准，通常不需要 re-encode。

### 3.6 本地 label_video.py（只执行 1 次！）

```bash
SCRIPT="D:/Desktop_Files/GPU-Train/RTX6000/Magicbot_Z1/scripts/label_video.py"
VIDDIR="D:/Desktop_Files/GPU-Train/RTX6000/Magicbot_Z1/videos"

# Isaac Lab 视频 (输入 fps=50, 输出 fps=50)
python "$SCRIPT" \
  "$VIDDIR/s1_flat/s1_flat_isaaclab_tmp.mp4" \
  -o "$VIDDIR/s1_flat/s1_flat_model3861_isaaclab.mp4" \
  --run s1_flat --model model_3861 \
  --time-out 0.9712 --ep-len 970 --bad-ori 0.0288 --vel-err 0.33

# MuJoCo 视频 (输入 fps=50, 输出 fps=50)
python "$SCRIPT" \
  "$VIDDIR/s1_flat/s1_flat_raw_mujoco.mp4" \
  -o "$VIDDIR/s1_flat/s1_flat_m3861_sim2sim_mujoco.mp4" \
  --run s1_flat --model model_3861 \
  --time-out 0.9712 --ep-len 970 --bad-ori 0.0288 --vel-err 0.33
```

> `label_video.py` 现在自动读取输入视频的 fps 并保留，不再硬编码 30fps。

### 3.7 清理临时文件

```bash
VIDDIR="D:/Desktop_Files/GPU-Train/RTX6000/Magicbot_Z1/videos"
find "$VIDDIR" -name '*_raw_*.mp4' -delete
find "$VIDDIR" -name '*_tmp.mp4' -delete
```

---

## 4. 当前 bestmodel 对照表

| Stage | Run Dir | Best Model | peak_reward | time_out | vel_err |
|-------|---------|------------|-------------|----------|---------|
| s1_flat | 2026-04-30_04-53-17_s1_flat | model_3861 | 48.35 | 97.1% | 0.33 m/s |
| s2_gentle | 2026-05-01_04-50-05_s2_gentle | model_47862 | 48.10 | 97.4% | 0.33 m/s |
| s3_rough_l2 | 2026-05-01_07-04-35_s3_rough_l2 | model_32790 | 38.94 | 94.1% | 0.41 m/s |
| s4_full_terrain | 2026-05-04_16-56-05_s4_full_terrain | model_8100 | 67.64 | 69.3% | 1.39 m/s |

JIT policy 路径：`logs/rsl_rl/magiclab_z1_12dof_velocity/<run_dir>/exported/policy.pt`

---

## 5. 视频输出目录

```
D:\Desktop_Files\GPU-Train\RTX6000\Magicbot_Z1\videos\
├── s1_flat\
│   ├── s1_flat_model3861_isaaclab.mp4       ← Isaac Lab, 50fps, 10s, labeled
│   └── s1_flat_m3861_sim2sim_mujoco.mp4     ← MuJoCo, 50fps, ~10s, labeled
│
├── s2_gentle\
│   ├── s2_gentle_model47862_isaaclab.mp4
│   └── s2_gentle_m47862_sim2sim_mujoco.mp4
│
├── s3_rough_l2\
│   ├── s3_rough_l2_model32790_isaaclab.mp4
│   └── s3_rough_l2_m32790_sim2sim_mujoco.mp4
│
├── s4_full_terrain\
│   ├── s4_full_terrain_model8100_isaaclab.mp4
│   └── s4_full_terrain_m8100_sim2sim_mujoco.mp4
│
└── sim2sim_experiments\                      ← MuJoCo 参数调优实验
```

---

## 6. 一键录制脚本（rtx_record_video.sh）

```bash
# 用法
bash rtx_record_video.sh <RUN_DIR> <CHECKPOINT> [VIDEO_LENGTH] [VEL_X] [DURATION]

# 示例
bash rtx_record_video.sh \
    magiclab_z1_12dof_velocity/2026-05-04_16-56-05_s4_full_terrain model_8100 500 0.5 10
```

自动流程：GPU 检查 → JIT 导出 → Isaac Sim 录制 → MuJoCo 录制 → 输出下载命令。

---

## 7. 验证清单

| 验证项 | 预期结果 | 状态 |
|--------|---------|------|
| `export_jit.py` 导出 | `exported/policy.pt` ~1.1 MB | ✅ |
| JIT = checkpoint 行为一致 | `empirical_normalization=False`, 无归一化差异 | ✅ |
| Isaac Sim 录制 fps=50 | 视频实时播放，无快放/慢放 | ✅ 2026-05-07 修复 |
| video_length=500 | 10s 模拟时间 = 10s 视频时长 | ✅ |
| `label_video.py` 保留 fps | 输出 fps = 输入 fps (不再硬编码30) | ✅ |
| Camera tracking | headless 模式下机器人可见 | ✅ |
| Renderer warmup | 无黑帧（20步预热） | ✅ |
| RTX MuJoCo EGL | 机器人站稳不摔倒 | ✅ |

---

## 8. 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 视频中机器人速度明显偏慢 | RecordVideo fps=30 ≠ 控制频率50Hz | `play_z1_video.py` 中传 `fps=50` |
| label 后视频变慢放 | `label_video.py` 硬编码 fps=30 | 已修复：自动读取输入 fps |
| 视频只有 4 秒太短 | video_length=200 (200×0.02=4s) | 改为 500 (10s) |
| Isaac Sim 视频 PyAV 读不了 | Isaac Sim 输出容器格式不标准 | 先 ffmpeg re-encode |
| 视频有两层 label | label_video.py 被执行了两次 | 只执行 1 次，对 re-encode 后的 tmp 文件 |
| RTX Isaac Sim 录制全黑 | 渲染器未预热 | 已有 20 步 warmup |
| RTX Isaac Sim 看不到机器人 | headless 模式摄像头停在原点 | 已有 camera tracking |
| MuJoCo 机器人摔倒 | 缺少 sim2sim 修正 | 使用 `mujoco_manual.py`（已内置修正） |
| `play_z1_video.py` 报 `terrain_generator` AttributeError | play config 访问 `terrain_generator.num_rows` 但为 None | 已加 None guard |
| `play_z1_video.py` 报 `Reward term not found` | curriculum 引用 weight=0 的 reward term | play config 中禁用 curriculum |
