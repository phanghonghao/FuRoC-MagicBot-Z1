# Z1 手动训练指令

> SSH 到 RTX 后直接复制粘贴运行，不依赖 Claude 代执行。

## 快速开始

```bash
ssh phh@192.168.120.155
cd ~/magiclab_rl_lab
source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
```

---

## 目录

| 文件 | 内容 |
|------|------|
| [launch_training.md](launch_training.md) | 启动训练（单卡 / 多卡 / 各阶段） |
| [monitor_and_control.md](monitor_and_control.md) | 监控、停止、查看状态 |
| [config_generation.md](config_generation.md) | 用 orchestrator 生成配置（不启动训练） |
| [video_recording.md](video_recording.md) | 录制 Isaac Sim / MuJoCo 视频 |
| [zombie_cleanup.md](zombie_cleanup.md) | 清理僵尸进程、GPU 占用 |
| [phase_params.md](phase_params.md) | 各阶段参数速查表 |
