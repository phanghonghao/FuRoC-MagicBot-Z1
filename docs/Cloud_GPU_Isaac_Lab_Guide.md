# 云GPU运行Isaac Lab仿真部署指南

> 使用场景：RTX 6000D 本地服务器运行 Isaac Lab 训练（MagicBot Z1 12DOF），本地电脑无 RTX GPU 无法运行 Isaac Sim。希望通过 Paratera 云 GPU 独立运行仿真进行可视化/录制。

---

## 1. 架构概览

```
┌──────────────────────────┐       ┌──────────────────┐       ┌──────────────────────────┐
│  RTX 6000D 内网服务器      │       │  你的本地电脑      │       │  Paratera 云GPU           │
│  192.168.120.155          │       │  (无 RTX GPU)     │       │  RTX 4090 / Linux容器      │
│                           │       │                  │       │                           │
│  Isaac Lab 训练 (Z1)      │──SCP──│  中转站           │──SCP──│  ???                      │
│  checkpoint 导出           │       │  下载→上传        │       │                           │
│  rsl_rl 训练日志           │       │                  │       │                           │
└──────────────────────────┘       └──────────────────┘       └──────────────────────────┘
        内网 IP                          你的电脑                    公网 IP
```

**核心约束**：RTX 6000D 是内网地址（192.168.120.155），云 GPU 无法直连。所有文件传输需通过本地电脑中转。

---

## 2. 关键发现：云容器跑不了 Isaac Sim

### 2.1 问题

在 Paratera **容器**（非云服务器 VM）上安装 Isaac Sim 后尝试运行，Vulkan 报错：

```
ERROR: [Loader Message] Code 0 : loader_scanned_icd_add: Could not get 'vkCreateInstance'
    via 'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0
ERROR: [Loader Message] Code 0 : vkCreateInstance: Found no drivers!
```

### 2.2 根本原因

**Isaac Lab 依赖 Isaac Sim，Isaac Sim 依赖 GPU 渲染管线。**

即使 `--headless` 模式，Isaac Sim（Omniverse Kit）也需要 GPU 的 **Vulkan/EGL 渲染能力**来生成画面。云容器的 NVIDIA 驱动**只暴露了 CUDA 计算能力**，没有暴露 Vulkan/OpenGL/EGL 渲染能力。

这是大多数云 GPU **容器**产品的通病——GPU 计算能用，但渲染管线被截断了。

### 2.3 容器 vs 云服务器

| | 容器 (Container) | 云服务器 (VM) |
|---|---|---|
| GPU 驱动 | 仅 CUDA 计算 | **完整驱动**（CUDA + Vulkan + EGL） |
| Isaac Sim | **不能跑** | 能跑 |
| MuJoCo EGL | 可能能跑 | 能跑 |
| RDP/VNC | WebSSH / noVNC | RDP（Windows）/ VNC（Linux） |
| 价格 | 便宜 | 稍贵 |

---

## 3. 可行方案

### 方案 A：在 RTX 6000 上直接录制（推荐，已验证）

RTX 6000 有完整 NVIDIA 驱动 + 物理显示器，Isaac Sim 和 MuJoCo 都能跑。

```bash
# 已有的一键录制脚本
bash D:/Desktop_Files/GPU-Train/RTX6000/rtx_record_video.sh <RUN_DIR> <CHECKPOINT>

# 或使用 gpu-train skill
/gpu-train --sim --best v4_gentle
```

**优点**：零额外成本，已验证能跑
**缺点**：需要 RTX 6000 在线，录制期间不能同时跑 Isaac Sim（会占所有 GPU）

### 方案 B：云容器跑 MuJoCo（待验证）

MuJoCo 的 EGL 渲染比 Isaac Sim 轻量得多，可能绕过 Vulkan 限制。

```bash
# 在云容器中尝试
export MUJOCO_GL=egl
python sim2sim/mujoco_manual.py \
    --mjcf ~/magicbot-z1_description/mjcf/MAGICBOTZ1.xml \
    --policy policy.pt \
    --record /tmp/z1_cloud.mp4 \
    --num_steps 500 --vel_x 0.5
```

**验证步骤**（SSH 到云容器后）：
```bash
# 1. 检查 CUDA 是否可用
nvidia-smi

# 2. 检查 EGL 是否可用
python -c "import mujoco; print(mujoco.__version__)"
python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('MAGICBOTZ1.xml')
d = mujoco.MjData(m)
mujoco.mj_step(m, d)
print('MuJoCo step OK')
"

# 3. 检查 EGL 渲染
python -c "
import os
os.environ['MUJOCO_GL'] = 'egl'
import mujoco
mujoco.MjModel.from_xml_path('MAGICBOTZ1.xml')
r = mujoco.Renderer(m, 640, 480)
r.render()
print('EGL render OK, shape:', r.render().shape)
"
```

**如果 EGL 通过** → MuJoCo 录制可行，不需要 Isaac Sim
**如果 EGL 也失败** → 回到方案 A 或 C

### 方案 C：换 Paratera 云服务器（VM，非容器）

在 Paratera 上创建**云服务器**（非容器），选择：
- GPU：RTX 4090
- 系统：**Windows Server** 或 **Ubuntu 桌面版**
- 存储：>= 200GB

云服务器是完整虚拟机，GPU 驱动完整（CUDA + Vulkan + EGL），Isaac Sim 可以正常运行。

**Windows VM + RDP 连接**：
```
Win + R → mstsc → 输入云服务器 IP:3389 → 连接
```

连上后就是完整 Windows 桌面，和操作本地电脑一样，Isaac Sim 渲染画面通过 RDP 实时传回。

### 方案对比

| | 方案 A (RTX6000) | 方案 B (容器+MuJoCo) | 方案 C (云服务器VM) |
|---|---|---|---|
| Isaac Sim 画面 | 有 | **无** | 有 |
| MuJoCo 画面 | 有 | 可能有 | 有 |
| 实时观看 | 需 SSH 到 RTX | 需 VNC | RDP 直连 |
| 额外成本 | 无 | 低 | 中 |
| 状态 | **已验证** | 待验证 | 待尝试 |

---

## 4. 文件传输（三步中转，所有方案通用）

由于 RTX 6000D 在内网，云 GPU 在公网，两者无法直接通信。文件需通过本地电脑中转。

### 4.1 从 RTX 6000D 下载到本地

```bash
# 在本地电脑执行（Git Bash / PowerShell）
mkdir -p D:/tmp/transfer

# 下载源码 + 脚本（~20MB）
scp -r phh@192.168.120.155:~/magiclab_rl_lab/source/ D:/tmp/transfer/source/
scp -r phh@192.168.120.155:~/magiclab_rl_lab/scripts/ D:/tmp/transfer/scripts/

# 下载 JIT policy（~7MB）
scp phh@192.168.120.155:~/magiclab_rl_lab/logs/rsl_rl/<RUN_DIR>/exported/policy.pt \
    D:/tmp/transfer/policy.pt

# 下载 MJCF 文件（MuJoCo 方案需要）
scp phh@192.168.120.155:~/magicbot-z1_description/mjcf/MAGICBOTZ1.xml D:/tmp/transfer/
```

### 4.2 从本地上传到云 GPU

```bash
# 上传到云容器/服务器
scp -r D:/tmp/transfer/ root@<云IP>:/root/shared-nvme/magiclab_rl_lab/
```

---

## 5. 待办（明天继续）

- [ ] 获取云 GPU 的 SSH 连接信息
- [ ] 验证方案 B：云容器 MuJoCo EGL 是否可用
- [ ] 如果方案 B 失败，评估方案 C（换成云服务器 VM）
- [ ] 根据 POC 结果编写 `cloud_record_video.sh` 自动化脚本

---

## 6. 云容器参数记录（供参考）

以下为 Paratera 容器默认参数，大部分不需要修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 镜像类型 | 公共镜像 | 可搜社区镜像 |
| 开发环境 | PyTorch | Isaac Sim 安装时会自带匹配版本 |
| GPU | RTX 4090 | 24GB VRAM |
| PyTorch 版本 | 2.7.0 | 基础镜像版本，不影响 Isaac Sim |
| 基础镜像 | PyTorch-25.03-py3 (Ubuntu 24.04) | 兼容 Isaac Sim 4.5+ |
| 自动挂载共享存储 | 开启 | 系统盘临时，共享存储持久化 |
| 共享存储路径 | `/root/shared-nvme` | 所有持久化数据放这里 |
| 共享存储容量 | 50GB 免费 | **需扩到 200GB+**（Isaac Sim 约 100GB+） |

> **注意**：如果走方案 B（MuJoCo only），不需要安装 Isaac Sim，共享存储 50GB 可能就够了（MuJoCo + 项目文件约 2-3GB）。
