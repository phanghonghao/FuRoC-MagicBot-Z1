# Z1 Humanoid Policy Viewer 集成指南

将 Z1 12-DOF 人形机器人的 velocity tracking policy 集成到 [humanoid-policy-viewer](https://github.com/Axellwppr/humanoid-policy-viewer)（基于 MuJoCo WASM + ONNX Runtime Web 的浏览器端策略可视化平台）。

---

## 目录

- [架构概览](#架构概览)
- [文件结构](#文件结构)
- [Policy 转换流程](#policy-转换流程)
- [观测空间定义](#观测空间定义)
- [MJCF 场景配置](#mjcf-场景配置)
- [PD 控制参数](#pd-控制参数)
- [前端集成细节](#前端集成细节)
- [Sim-to-Sim 注意事项](#sim-to-sim-注意事项)
- [常见问题排查](#常见问题排查)

---

## 架构概览

```
┌─────────────────────────────────────────────────────┐
│                   Browser (WASM)                     │
│                                                      │
│  ┌──────────┐   ┌──────────────┐   ┌─────────────┐ │
│  │ MuJoCo   │──▶│ Observation  │──▶│  ONNX       │ │
│  │ WASM     │   │ Pipeline     │   │  Runtime    │ │
│  │ (z1.xml) │   │ (47dim×5=235)│   │  (.onnx)    │ │
│  └──────────┘   └──────────────┘   └──────┬──────┘ │
│       ▲                                    │        │
│       │          action (12dim)            │        │
│       │         × action_scale(0.25)       │        │
│       │         + default_joint_pos        │        │
│       │         → PD torque → ctrl         │        │
│       └────────────────────────────────────┘        │
│                                                      │
│  UI: Velocity Command Sliders (vx, vy, vyaw)        │
└─────────────────────────────────────────────────────┘
```

---

## 文件结构

```
humanoid-policy-viewer/
├── public/examples/
│   ├── scenes/
│   │   ├── files.json                    # 场景文件索引（已添加 Z1 条目）
│   │   └── z1/
│   │       ├── z1.xml                    # Z1 MJCF 场景文件
│   │       └── meshes/                   # 25个 STL 网格文件
│   │           ├── pelvis.STL
│   │           ├── LINK_HIP_PITCH_L.STL
│   │           └── ...
│   └── checkpoints/
│       └── z1/
│           ├── p2_fine_policy.onnx       # ONNX 推理模型
│           └── velocity_policy.json      # 策略配置文件
├── scripts/
│   └── export_z1_onnx.py                # TorchScript → ONNX 转换脚本
└── src/
    ├── simulation/
    │   ├── main.js                       # 主仿真循环 + PD 控制
    │   ├── mujocoUtils.js                # MuJoCo 工具函数
    │   └── observationHelpers.js         # 观测计算类
    └── views/
        └── Demo.vue                      # 前端 UI 组件
```

---

## Policy 转换流程

### 1. TorchScript → ONNX

使用 `scripts/export_z1_onnx.py`：

```bash
cd humanoid-policy-viewer
python scripts/export_z1_onnx.py
```

关键参数：
- **obs_dim = 235**（47 per frame × 5 history frames）
- 使用 `dynamo=False`（legacy TorchScript exporter），因为 PyTorch 2.9+ 的新 ONNX exporter 不兼容 TorchScript

```python
torch.onnx.export(
    model,
    torch.randn(1, 235),
    output_path,
    input_names=["policy"],
    output_names=["action"],
    dynamo=False  # 必须用 legacy exporter
)
```

### 2. 策略配置文件

`velocity_policy.json` 定义了模型路径、观测配置、PD 参数等：

```json
{
  "onnx": {
    "path": "./examples/checkpoints/z1/p2_fine_policy.onnx",
    "meta": {
      "in_keys": ["policy"],
      "out_keys": ["action"],
      "in_shapes": [[[1, 235]]]
    }
  },
  "obs_config": {
    "policy": [
      {
        "name": "VelocityPolicyObs",
        "history_length": 5,
        "period": 0.6
      }
    ]
  },
  "policy_joint_names": [ ... ],
  "action_scale": [0.25, ...],
  "stiffness": [100,100,100,150,60,60, 100,100,100,150,60,60],
  "damping": [4,4,4,5,3,3, 4,4,4,5,3,3],
  "default_joint_pos": [-0.35,0,0,0.7,-0.35,0, -0.35,0,0,0.7,-0.35,0]
}
```

---

## 观测空间定义

### 单帧观测 (47 维)

| 索引 | 维度 | 名称 | 说明 |
|------|------|------|------|
| 0-2 | 3 | base_ang_vel | 基座角速度（**体坐标系**） |
| 3-5 | 3 | projected_gravity | 重力投影向量 |
| 6-8 | 3 | velocity_commands | 速度指令 (vx, vy, vyaw) |
| 9-20 | 12 | joint_pos_rel | 关节位置 - 默认位置 |
| 21-32 | 12 | joint_vel | 关节角速度 |
| 33-44 | 12 | last_action | 上一步动作 |
| 45-46 | 2 | gait_phase | 步态相位 (sin, cos) |

### 历史堆叠 (5帧 × 47 = 235 维)

RSL-RL 使用 `history_length=5`，将最近 5 帧观测拼接：

```
[obs(t-4), obs(t-3), obs(t-2), obs(t-1), obs(t)] → 235 维
```

实现：`VelocityPolicyObs` 类（`observationHelpers.js`）

### 关键坐标系转换

MuJoCo free joint 的 `qvel[3:6]` 是**世界坐标系**角速度，而 Isaac Lab 的 `base_ang_vel` 是**体坐标系**。需要在 `main.js` 中做转换：

```javascript
// 四元数逆旋转: q^-1 * v_world = v_body
const qix = -qx, qiy = -qy, qiz = -qz, qiw = qw;
// v + 2*qw*(qi × v) + 2*(qi × (qi × v))
```

---

## MJCF 场景配置

### 关键修改点

原始 MJCF 来自 `magicbot-z1_description/mjcf/MAGICBOTZ1.xml`，以下为 viewer 适配的修改：

1. **meshdir 路径**：`meshdir="../meshes/"` → `meshdir="meshes/"`
2. **关节阻尼**：`damping="10"` → `damping="2.0"`（配合显式 PD 控制）
3. **执行器力矩限制**：

```xml
<motor name="left_hip_pitch_actuator" joint="JOINT_HIP_PITCH_L"
       ctrllimited="true" ctrlrange="-200 200"/>
```

| 关节 | ctrlrange (Nm) |
|------|----------------|
| hip_pitch / hip_roll | ±200 |
| hip_yaw | ±150 |
| knee | ±200 |
| ankle_pitch / ankle_roll | ±100 |

> **必须** 设置 `ctrllimited="true"` 和 `ctrlrange`，否则显式 PD 控制器会产生无界力矩，导致 MuJoCo NaN 爆炸。

### 脚部几何

原始 MJCF 的脚部 mesh 被注释掉，替换为 box 碰撞体以提升稳定性：

```xml
<geom name="l_foot" type="box" size="0.09 0.04 0.005" pos="0.03 0.0 -0.01" group="2"/>
```

---

## PD 控制参数

Viewer 使用**显式 PD 控制**，而非 Isaac Lab 的隐式 PD：

```
torque = kp × (target_pos - current_pos) + kd × (0 - current_vel)
```

| 关节 | kp (stiffness) | kd (damping) |
|------|----------------|--------------|
| hip_pitch | 100 | 4 |
| hip_roll | 100 | 4 |
| hip_yaw | 100 | 4 |
| knee | 150 | 5 |
| ankle_pitch | 60 | 3 |
| ankle_roll | 60 | 3 |

目标位置计算：
```javascript
target_pos = default_joint_pos + action × action_scale(0.25)
```

---

## 前端集成细节

### 速度指令 UI (Demo.vue)

三个滑块控制速度指令：
- **Forward (vx)**: -1.0 ~ 2.0 m/s
- **Lateral (vy)**: -1.0 ~ 1.0 m/s
- **Yaw (vyaw)**: -3.0 ~ 3.0 rad/s

### 初始关节位置

仿真重置时设置默认站立姿态，防止零位（直腿）导致的初始不稳定：

```javascript
// default_joint_pos: [-0.35, 0, 0, 0.7, -0.35, 0, -0.35, 0, 0, 0.7, -0.35, 0]
// 对应: hip_pitch=-0.35, knee=0.7, ankle_pitch=-0.35 (微蹲姿态)
```

### 关节名称映射

| Isaac Lab (URDF) | MuJoCo (MJCF) |
|-------------------|----------------|
| left_hip_pitch_joint | JOINT_HIP_PITCH_L |
| left_hip_roll_joint | JOINT_HIP_ROLL_L |
| left_hip_yaw_joint | JOINT_HIP_YAW_L |
| left_knee_joint | JOINT_KNEE_PITCH_L |
| left_ankle_pitch_joint | JOINT_ANKLE_PITCH_L |
| left_ankle_roll_joint | JOINT_ANKLE_ROLL_L |
| right_* (同理) | *_R |

---

## Sim-to-Sim 注意事项

### Isaac Lab vs MuJoCo 差异

| 特性 | Isaac Lab (PhysX) | MuJoCo WASM |
|------|-------------------|-------------|
| PD 控制 | 隐式（PhysX 内置） | 显式（JS 计算 torque） |
| 角速度帧 | 体坐标系 | 世界坐标系（需转换） |
| 时间步长 | ~50Hz (DT=0.02) | 0.002s (500Hz) |
| 关节阻尼 | PD 控制器隐含 | 需要显式设置 |
| 力矩限制 | URDF effort | MJCF ctrlrange |

### 已知的 Sim-to-Sim Gap

1. **接触模型差异**：PhysX 和 MuJoCo 的接触力学不同，可能影响脚部抓地力
2. **阻尼行为**：显式 PD 的阻尼效果与隐式 PD 不完全等价
3. **数值精度**：WASM 单精度浮点 vs GPU 训练时的混合精度

---

## 常见问题排查

### Q: NaN 爆炸（QACC warning）

```
WARNING: Nan, Inf or huge value in QACC at DOF X
```

**原因**：执行器无 `ctrlrange`，PD 力矩无限大
**解决**：在 MJCF 的 `<motor>` 上添加 `ctrllimited="true" ctrlrange="-200 200"`

### Q: 机器人立刻摔倒

检查以下几点：
1. 初始关节位置是否正确设置（`default_joint_pos`）
2. 角速度是否做了世界→体坐标系转换
3. 关节阻尼是否合理（推荐 1.0 ~ 3.0）
4. ONNX 输入维度是否为 235（47×5）

### Q: ONNX 导出失败 `dynamo` 相关错误

PyTorch 2.9+ 新 ONNX exporter 不支持 TorchScript，需要 `dynamo=False`：

```python
torch.onnx.export(model, ..., dynamo=False)
```

### Q: 模型 shape mismatch (1x47 vs 235x512)

`history_length=5`，实际输入是 47×5=235 维，不是 47 维。

---

## 启动方式

```bash
cd humanoid-policy-viewer
npm install   # 首次运行
npm run dev   # 启动开发服务器
```

访问 **http://localhost:3000** 即可看到 Z1 策略可视化。
