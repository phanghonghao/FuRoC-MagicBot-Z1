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
| Up | 前进加速 | 0 → 1.0 m/s | +0.1 |
| Down | 后退加速 | 0 → -0.5 m/s | -0.1 |
| Left | 左转 | 0 → -0.5 rad/s | -0.1 |
| Right | 右转 | 0 → 0.5 rad/s | +0.1 |
| Q | 左侧移 | 0 → 0.5 m/s | +0.1 |
| E | 右侧移 | 0 → -0.5 m/s | -0.1 |
| Space | 停止（归零） | — | — |
| Esc | 退出 | — | — |
| Tab | 折叠/展开左侧面板 | — | — |

> 使用方向键而非 WASD，避免与 MuJoCo Viewer 内置的相机/灯光快捷键冲突。

> 速度范围与训练一致: `lin_vel_x=[-0.5, 1.0]`, `lin_vel_y=[-0.5, 0.5]`, `ang_vel_z=[-0.5, 0.5]`

---

## 3. Viewer 面板说明

MuJoCo Viewer 默认三栏布局：**左面板 | 3D 视图 | 右面板**

### 面板折叠

| 操作 | 方式 |
|------|------|
| 折叠/展开左侧面板 | 按 **Tab** 键 |
| 折叠右侧面板 | 点击右侧面板标题栏上的 **折叠箭头** |
| 展开 | 再次点击折叠箭头 |

### 面板内容

**左侧面板** — 渲染与仿真控制：
- 渲染选项（线框、阴影、透明度等）
- 仿真速度控制
- 几何体组开关（显示/隐藏碰撞体、关节轴等）

**右侧面板** — 关节与执行器数据（重点关注）：

| 区域 | 内容 | 关注点 |
|------|------|--------|
| **Joint** | 各关节角度 (qpos) | 关节是否达到目标位置 |
| **Joint** | 各关节角速度 (qvel) | 运动是否平滑 |
| **Control** | 各执行器输出 (ctrl) | 即 PD 控制器输出的力矩 |
| **Sensor** | 传感器数据 | IMU 角速度、姿态等 |

> 右侧面板的 **Control** 区域显示的就是关节力矩（单位 N·m），即 PD 控制器输出 `τ = kp × (q_des - q) - kd × q̇` 的值。如果力矩持续达到限幅（120 N·m for hip/knee, 50 N·m for ankle），说明关节饱和。

### 3D 视图控制

| 操作 | 方式 |
|------|------|
| 旋转视角 | 鼠标左键拖拽 |
| 平移视角 | 鼠标右键拖拽 |
| 缩放 | 鼠标滚轮 |
| 恢复视角 | 双击鼠标左键 |

---

## 4. 训练指标查看

MuJoCo Viewer 只显示物理仿真状态（关节、力矩、传感器），**不包含训练指标**。

训练指标（reward、time_out、entropy 等）的查看方式：

| 指标 | 查看方式 |
|------|----------|
| 实时训练指标 | `/gpu-train --tail`（从 RTX 服务器读取日志） |
| 训练趋势分析 | `/gpu-train --check` |
| 过拟合检测 | `/gpu-train --monitor` |
| 学习曲线图表 | `/plot-train-Z1` |
| 历史训练记录 | `docs/training_logs/training_log_*.md` |
| Phase 进度总览 | `docs/bestmodel_phase.json` |

---

## 5. 显式 PD 说明

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

`default_joint_pos` 保持硬编码（来源于 `magiclab_rl_lab/source/.../assets/robots/magiclab.py` 中的 `IdealPDActuatorCfg`，按 MuJoCo joint 顺序排列）。deploy.yaml 导出的 joint 顺序与 MuJoCo 不匹配，不从 deploy.yaml 读取。

---

## 6. 相位感知地形映射

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

## 7. 参数说明

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

## 8. 常用指令

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

## 9. 文件位置

| 文件 | 路径 | 用途 |
|------|------|------|
| 本地部署 | `Magicbot_Z1\sim2sim\mujoco_manual.py` | 本地键盘测试 |
| RTX 镜像 | `Magicbot_Z1\magiclab_rl_lab\sim2sim\mujoco_manual.py` | 远程录制 |
| MJCF 模型 | `magicbot-z1_description\mjcf\MAGICBOTZ1.xml` | 机器人模型 |

---

## 10. 依赖

| 依赖 | 用途 |
|------|------|
| MuJoCo 3.8.0 | 物理仿真 + Viewer |
| PyTorch | JIT policy 加载 |
| GLFW | MuJoCo Viewer 渲染后端 |
| imageio / ffmpeg | 仅录制视频需要 |
