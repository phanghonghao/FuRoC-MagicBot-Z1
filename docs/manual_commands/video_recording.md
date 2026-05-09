# 录制视频

> MuJoCo 可以在训练时录制（2 min），Isaac Sim 必须在训练停止时录制（15-20 min）。

## 前提：导出 JIT Policy

```bash
CHECKPOINT="logs/rsl_rl/magiclab_z1_12dof_velocity/<RUN_DIR>/model_<N>.pt"

python -u scripts/export_jit.py --checkpoint ${CHECKPOINT}
# 输出: <RUN_DIR>/exported/policy.pt
```

## MuJoCo 录制（训练时可并行）

```bash
POLICY="logs/rsl_rl/magiclab_z1_12dof_velocity/<RUN_DIR>/exported/policy.pt"
SAVE_NAME="p3b_coarse_model10500"

python -u sim2sim/mujoco_manual.py \
    --mjcf ~/magicbot-z1_description/mjcf/MAGICBOTZ1.xml \
    --policy ${POLICY} \
    --record /tmp/${SAVE_NAME}_mujoco.mp4 \
    --num_steps 500 \
    --vel_x 0.5
```

下载到本地：
```bash
# 在本地 Windows 执行
scp phh@192.168.120.155:/tmp/p3b_coarse_model10500_mujoco.mp4 "D:/Desktop_Files/GPU-Train/RTX6000/Magicbot_Z1/videos/p/p3b_coarse/"
```

## Isaac Sim 录制（必须停止训练）

```bash
CHECKPOINT="logs/rsl_rl/magiclab_z1_12dof_velocity/<RUN_DIR>/model_<N>.pt"

python -u scripts/rsl_rl/play_z1_video.py \
    --checkpoint ${CHECKPOINT} \
    --video \
    --video_length 400 \
    --headless \
    --num_envs 16 \
    --device cuda:0
```

视频输出：`<RUN_DIR>/videos/play/rl-video-step-0.mp4`

下载到本地：
```bash
scp phh@192.168.120.155:~/magiclab_rl_lab/logs/rsl_rl/magiclab_z1_12dof_velocity/<RUN_DIR>/videos/play/rl-video-step-0.mp4 \
    "D:/Desktop_Files/GPU-Train/RTX6000/Magicbot_Z1/videos/p/p3b_coarse/p3b_coarse_isaaclab.mp4"
```

## 下载训练参数（随视频一起保存）

```bash
# 本地 Windows 执行
scp -r phh@192.168.120.155:~/magiclab_rl_lab/logs/rsl_rl/magiclab_z1_12dof_velocity/<RUN_DIR>/params/ \
    "D:/Desktop_Files/GPU-Train/RTX6000/Magicbot_Z1/videos/p/p3b_coarse/params/"
```

## 加标签（本地执行）

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
