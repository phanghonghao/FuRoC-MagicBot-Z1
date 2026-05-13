# BridgeDP Z1 部署指南 — 学舞蹈

> 平台地址: https://engine.bridgedp.com/workspace/operation

---

## Z1 硬件信息

- 主控: NVIDIA Jetson Orin NX
- 架构: **ARM64 (aarch64)**

---

## 第一步：确认 BridgeDP 是否支持 Z1

在 Z1 机器人上执行：

```bash
# 试探是否有 Z1 专用包
wget --spider "https://bridgedp-platform-public.oss-cn-shenzhen.aliyuncs.com/ota-pkg/robot_ota_system_node_Z1_arm_release.tar.gz"
```

- 返回 **200** → 有 Z1 专用包，使用该包
- 返回 **404** → 无专用包，尝试 G1 ARM 版（架构相同）

---

## 第二步：下载安装包

### 方案 A：有 Z1 专用包

```bash
wget -O ./bridgedp_robot_ota_release.tar.gz "https://bridgedp-platform-public.oss-cn-shenzhen.aliyuncs.com/ota-pkg/robot_ota_system_node_Z1_arm_release.tar.gz"
```

### 方案 B：无 Z1 包，用 G1 ARM 版（架构相同，但可能有型号校验）

```bash
wget -O ./bridgedp_robot_ota_release.tar.gz "https://bridgedp-platform-public.oss-cn-shenzhen.aliyuncs.com/ota-pkg/robot_ota_system_node_G1_arm_release.tar.gz"
```

---

## 第三步：安装

```bash
# 删除旧凭证文件
sudo rm ~/.robot_credential

# 解压并进入目录
mkdir -p bridgedp_robot_ota_release
tar -zxvf ./bridgedp_robot_ota_release.tar.gz -C bridgedp_robot_ota_release --strip-components=1
cd bridgedp_robot_ota_release

# 安装系统服务
sudo bash ./install_ota_service.sh
sudo bash ./install_robot_control_service.sh
sudo systemctl enable bridgedp-platform-ota-root.service

# 交互式填写验证信息并启动
./run.sh -e
sudo systemctl restart bridgedp-platform-ota-root.service
```

---

## 第四步：平台端添加机器人

服务启动成功后，访问平台：

> https://engine.bridgedp.com/workspace/operation

在部署台「手动添加机器人」，填写 Z1 的验证信息。

---

## 注意事项

- 如果 `install_ota_service.sh` 或 `./run.sh -e` 报型号不匹配错误，说明 BridgeDP 目前仅支持 G1/T1，需要联系平台方添加 Z1 支持
- 安装脚本内容可提前检查：`cat ./install_ota_service.sh | grep -i "model\|robot\|G1\|T1\|Z1"`，看是否有型号白名单
- Z1 是 ARM 架构（Orin NX），与 G1 相同架构，理论上 ARM 包可以运行，但可能存在硬件驱动/接口差异
