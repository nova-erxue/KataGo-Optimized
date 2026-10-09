# Linux 从零安装环境

自动安装支持 **Ubuntu 22.04 / 24.04 x86_64**。这是脚本支持范围；引擎源码本身不检查 Linux 发行版。

已在提供的 Ubuntu 22.04 / RTX 5090 镜像上完成安装、重复执行、完整源码编译和模型搜索测试。Ubuntu 24.04 及独立主机驱动安装尚未实机验证。

## 已有 NVIDIA 驱动的云主机

先确认 `nvidia-smi` 能看到 RTX 50。在发布包目录执行：

```bash
sudo bash setup-linux.sh
source /opt/katago-rtx50/env.sh
./run-katago.sh version
```

脚本安装 CUDA 13.2 编译器和开发库、cuDNN 9.24.x、CMake、GCC/G++、Ninja、Git、zlib/libzip、Python 3 和 PyYAML。已有更高版本的 cuDNN 会保留，不自动降级。CUDA 13.2 与原有工具链可同时存在。旧 cuDNN 若被 APT 锁定，脚本只临时解除目标 cuDNN 包的锁定，安装后恢复。

只使用预编译版可用 `sudo bash setup-linux.sh --runtime-only`，省去完整 Toolkit 和编译工具。它仍安装 CUDA 13.2 的 cuBLAS、NVRTC、CUDA runtime 和 cuDNN。

脚本会添加 NVIDIA 官方 APT 仓库并安装系统包，需要网络和 root/sudo 权限。完整安装需要数 GB 下载和磁盘空间。没有 `sudo` 的 root 容器直接用 `bash`。不下载模型，不修改 iKataGo 配置，不启动联网服务，也不自动重启。

## 全新独立机器，尚未安装显卡驱动

```bash
sudo bash setup-linux.sh --install-driver
# 驱动安装结束后，自己重启电脑；启用 Secure Boot 时按系统提示完成 MOK 登记。
sudo reboot
# 重启后回到发布包目录
sudo bash setup-linux.sh
source /opt/katago-rtx50/env.sh
```

此选项安装 NVIDIA open 驱动和当前内核头文件。驱动安装阶段成功后以退出码 10 提醒重启，重启后再次执行才能完成 GPU 验证。驱动安装分支未在本次云容器内实测；脚本会拒绝在容器/WSL 内使用此选项。

云容器或 WSL 的驱动由宿主系统提供，不应在容器内安装。云平台未暴露 GPU、主机驱动过旧或不支持当前内核时，应先处理宿主系统。

## 检查、重试与路径

```bash
bash setup-linux.sh --dry-run             # 只看安装计划
bash setup-linux.sh --check               # 检查完整环境
bash setup-linux.sh --runtime-only --check # 只检查运行环境
```

安装中断后可以重复运行同一命令。已装的软件包由 APT 判断是否还需安装。日志、版本记录和 `env.sh` 默认保存在 `/opt/katago-rtx50`；可用 `--prefix /path/to/env` 更换这些文件的保存位置，APT 安装位置仍由系统决定。

每个新终端执行 `source /opt/katago-rtx50/env.sh`。CUDA 运行库会随环境切换；系统 cuDNN 被更新，已运行的其他 GPU 程序需要重新启动才能使用新库。iKataGo 安装时可显式指定：

```bash
python3 install_ikatago.py --work /root/work --cuda-root /usr/local/cuda-13.2
```

这样生成的 iKataGo 启动器会记录 CUDA 13.2 路径。运行前仍需准备原始 TF 权重；命令见 README。

安装方法依据 [NVIDIA CUDA Linux 安装文档](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) 和 [cuDNN Linux 安装文档](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/linux.html)。其他发行版请按官方文档准备依赖，再使用 BUILD.md 的编译入口。
