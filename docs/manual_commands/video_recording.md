# 录制视频

> 推荐方式：直接使用 `D:/Desktop_Files/GPU-Train/RTX6000/rtx_record_video.sh`。
> 当前一键流程会在 RTX 远端录制 Isaac Lab + MuJoCo，然后自动 `scp` 回本地、自动打标签、并删除本地 raw 视频。

> MuJoCo 可以在训练时录制（2 min），Isaac Sim 必须在训练停止时录制（15-20 min）。

## 前提：导出 JIT Policy

```bash
CHECKPOINT="logs/rsl_rl/magiclab_z1_12dof_velocity/<RUN_DIR>/model_<N>.pt"

python -u scripts/export_jit.py --checkpoint ${CHECKPOINT}
# 输出: <RUN_DIR>/exported/policy.pt
```

## MuJoCo 录制（旧手动方式，仅保留参考）

```bash
POLICY="logs/rsl_rl/magiclab_z1_12dof_velocity/<RUN_DIR>/exported/policy.pt"
SAVE_NAME="p3b_coarse_model10500"

python -u sim2sim/mujoco_manual.py \
    --mjcf ~/magicbot-z1_description/mjcf/MAGICBOTZ1.xml \
    --policy ${POLICY} \
    --record /tmp/${SAVE_NAME}_mujoco.mp4 \
    --num_steps 1000 \
    --vel_x 0.3
```

下载到本地（旧手动方式）：
```bash
# 在本地 Windows 执行
scp phh@192.168.120.155:/tmp/p3b_coarse_model10500_mujoco.mp4 "D:/Desktop_Files/GPU-Train/RTX6000/Magicbot_Z1/videos/p/p3b_coarse/"
```

## Isaac Sim 录制（旧手动方式，必须停止训练）

```bash
CHECKPOINT="logs/rsl_rl/magiclab_z1_12dof_velocity/<RUN_DIR>/model_<N>.pt"

python -u scripts/rsl_rl/play_z1_video.py \
    --checkpoint ${CHECKPOINT} \
    --video \
    --video_length 1000 \
    --headless \
    --num_envs 1 \
    --device cuda:<IDLE_GPU>
```

视频输出：`<RUN_DIR>/videos/play/rl-video-step-0.mp4`

下载到本地（旧手动方式）：
```bash
scp phh@192.168.120.155:~/magiclab_rl_lab/logs/rsl_rl/magiclab_z1_12dof_velocity/<RUN_DIR>/videos/play/rl-video-step-0.mp4 \
    "D:/Desktop_Files/GPU-Train/RTX6000/Magicbot_Z1/videos/p/p3b_coarse/p3b_coarse_isaaclab.mp4"
```

## 下载训练参数（旧手动方式）

```bash
# 本地 Windows 执行
scp -r phh@192.168.120.155:~/magiclab_rl_lab/logs/rsl_rl/magiclab_z1_12dof_velocity/<RUN_DIR>/params/ \
    "D:/Desktop_Files/GPU-Train/RTX6000/Magicbot_Z1/videos/p/p3b_coarse/params/"
```

## 加标签（旧手动方式，本地执行）

```bash
LABEL_SCRIPT="D:/Desktop_Files/GPU-Train/RTX6000/Magicbot_Z1/scripts/label_video.py"
VIDEO="D:/Desktop_Files/GPU-Train/RTX6000/Magicbot_Z1/videos/p/p3b_coarse/p3b_coarse_mujoco.mp4"

python "$LABEL_SCRIPT" "$VIDEO" \
    --model model_10500 \
    --run p3b_coarse \
    --reward 28.53 \
    --terrain intermediate \
    --iteration 10500 \
    --action-mean 0.65

# 覆盖原文件
mv "${VIDEO%.mp4}_labeled.mp4" "$VIDEO"
```

## 一键方式（推荐）

```bash
bash D:/Desktop_Files/GPU-Train/RTX6000/rtx_record_video.sh <RUN_DIR> <CHECKPOINT> [VIDEO_LENGTH] [VEL_X] [DURATION] [GPU_ID]
```

默认值：

- `VIDEO_LENGTH=1000`
- `VEL_X=0.3`
- `DURATION=20`
- `GPU_ID=auto`（自动选满足 `--idle` 规则的空闲卡）
