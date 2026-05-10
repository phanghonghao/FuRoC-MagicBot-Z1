# 本地 MuJoCo 键盘控制指南

> 环境: Windows 11, MuJoCo 3.8.0, Python 3.14

---

## 1. 快速启动

```powershell
cd D:\Desktop_Files\GPU-Train\RTX6000\Magicbot_Z1

# 推荐方式 — 通过 gpu-train skill
/gpu-train --local_play p2_fine

# 手动方式
python sim2sim\mujoco_manual.py --mjcf magicbot-z1_description\mjcf\MAGICBOTZ1.xml --policy models\p\p2_fine\p2_fine_policy.pt --deploy_cfg videos\p\p2_fine\params\deploy.yaml --keyboard --num_steps 10000
```

---

## 2. 键盘操作

按键直接在 **MuJoCo Viewer 窗口** 内生效（通过 GLFW key_callback），无需切换窗口焦点。

| 按键 | 功能 | 速度范围 | 步长 |
|------|------|----------|------|
| W | 前进加速 | 0 → 1.0 m/s | +0.1 |
| S | 后退加速 | 0 → -0.5 m/s | -0.1 |
| A | 左转 | 0 → -0.5 rad/s | -0.1 |
| D | 右转 | 0 → 0.5 rad/s | +0.1 |
| Q | 左侧移 | 0 → 0.5 m/s | +0.1 |
| E | 右侧移 | 0 → -0.5 m/s | -0.1 |
| Space | 停止（归零） | — | — |
| Esc | 退出 | — | — |

> 速度范围与训练一致: `lin_vel_x=[-0.5, 1.0]`, `lin_vel_y=[-0.5, 0.5]`, `ang_vel_z=[-0.5, 0.5]`

---

## 3. 显式 PD 说明

训练环境使用 `IdealPDActuatorCfg`（显式 PD），MuJoCo 部署也是显式 PD，公式完全一致：

```
τ = kp × (q_des - q) - kd × q̇
```

直接使用 deploy.yaml 中的 KP/KD 值，无需 sim2sim boost。

| 参数 | 训练值 (magiclab.py) | deploy.yaml | sim2sim 用值 |
|------|---------------------|-------------|-------------|
| Kp (hip) | 100.0 | 100.0 | 100.0 |
| Kp (knee) | 150.0 | 150.0 | 150.0 |
| Kp (ankle) | 60.0 | 60.0 | 60.0 |
| Kd (hip) | 4.0 | 4.0 | 4.0 |
| Kd (knee) | 5.0 | 5.0 | 5.0 |
| Kd (ankle) | 3.0 | 3.0 | 3.0 |

`default_joint_pos` 保持硬编码（deploy.yaml 导出的 joint 顺序与 MuJoCo 不匹配）。

---

## 4. 相位感知地形映射

```python
PHASE_TERRAIN = {
    "p1": None,       # flat ground
    "p2": None,       # flat ground
    "p3": "p3",       # gentle terrain
    "p3b": "p3b",     # intermediate terrain
    "p4": "p3b",      # rough → use p3b for sim2sim
}
```

优先级: `--terrain` > `--phase` > flat ground（默认）

---

## 5. 参数说明

```
--mjcf         (必填) MAGICBOTZ1.xml 路径
--policy       (必填) policy 文件路径 (.pt 或 .onnx)
--deploy_cfg   (可选) deploy.yaml 路径，提供 KP/KD/action_scale/action_offset
--onnx         使用 ONNX 模型（默认 JIT）
--vel_x        初速度 x (默认 0.5 m/s)
--vel_y        初速度 y (默认 0.0 m/s)
--vel_yaw      初角速度 (默认 0.0 rad/s)
--keyboard     启用键盘实时控制
--num_steps    运行步数 (默认 10000, 50Hz 下 = 200 秒)
--record       录制视频路径
--terrain      地形类型: "p3" 或 "p3b"
--phase        Phase ID: p1/p2/p3/p3b/p4 (自动选择地形)
--show_viewer  显示 MuJoCo viewer (默认开启)
```

---

## 6. 常用指令

### 指定 phase 自动加载地形

```powershell
# p3 → 自动加载 gentle terrain
python sim2sim\mujoco_manual.py --mjcf magicbot-z1_description\mjcf\MAGICBOTZ1.xml --policy models\p\p3_fine\p3_fine_policy.pt --deploy_cfg videos\p\p3_fine\params\deploy.yaml --phase p3 --keyboard --num_steps 10000

# p3b → 自动加载 intermediate terrain
python sim2sim\mujoco_manual.py --mjcf magicbot-z1_description\mjcf\MAGICBOTZ1.xml --policy models\p\p3b_fine\p3b_fine_policy.pt --deploy_cfg videos\p\p3b_fine\params\deploy.yaml --phase p3b --keyboard --num_steps 10000
```

### 录制视频

```powershell
python sim2sim\mujoco_manual.py --mjcf magicbot-z1_description\mjcf\MAGICBOTZ1.xml --policy models\p\p3b_fine\p3b_fine_policy.pt --deploy_cfg videos\p\p3b_fine\params\deploy.yaml --phase p3b --record output.mp4 --num_steps 500
```

### 自定义初速度

```powershell
python sim2sim\mujoco_manual.py --mjcf magicbot-z1_description\mjcf\MAGICBOTZ1.xml --policy models\p\p2_fine\p2_fine_policy.pt --deploy_cfg videos\p\p2_fine\params\deploy.yaml --keyboard --vel_x 1.0 --num_steps 10000
```

### 上传到 RTX 服务器

```powershell
scp D:\Desktop_Files\GPU-Train\RTX6000\Magicbot_Z1\magiclab_rl_lab\sim2sim\mujoco_manual.py phh@192.168.120.155:~/magiclab_rl_lab/sim2sim/mujoco_manual.py
```

---

## 7. 文件位置

| 文件 | 路径 | 用途 |
|------|------|------|
| 本地部署 | `Magicbot_Z1\sim2sim\mujoco_manual.py` | 本地键盘测试 |
| RTX 镜像 | `Magicbot_Z1\magiclab_rl_lab\sim2sim\mujoco_manual.py` | 远程录制 |
| MJCF 模型 | `magicbot-z1_description\mjcf\MAGICBOTZ1.xml` | 机器人模型 |

---

## 8. 依赖

| 依赖 | 用途 |
|------|------|
| MuJoCo 3.8.0 | 物理仿真 + Viewer |
| PyTorch | JIT policy 加载 |
| GLFW | MuJoCo Viewer 渲染后端 |
| imageio / ffmpeg | 仅录制视频需要 |
