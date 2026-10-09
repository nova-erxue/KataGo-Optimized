# KataGo 优化版引擎 — SM120 / CUDA 13.2

项目统一命名为 KataGo 优化版引擎。本次附件仅支持 SM120；后续其他 GPU 架构版本在同一仓库发布。

本次更新：显卡检查改为计算能力 12.0（SM120），支持 RTX PRO 6000 Blackwell 等 SM120 GPU；Linux 和 Windows 均重新编译。发布附件改用 SM120 名称，安装目录及 iKataGo 引擎配置名保留兼容。

KataGo 1.18.2 的第三方 RTX 50 优化分支，直接读取兼容的原始 TF/NBT 权重，保持 FP16 精度等级。

- 提供 Linux x86_64 和 Windows x64 预编译引擎，均用 CUDA 13.2 构建。
- Linux 包带独立 libzip 运行库与启动脚本，支持原版镜像部署。
- Linux 整合包新增全新部署脚本和三份 Notebook（katago / ikatago install / ikatago start）；iKataGo Server 5.0.1、穿透配置及公共平台令牌在用户机器上从上游下载，不附带于发布包；自动生成配置、交互设置账号，附启停管理。模型由用户提供，远程发现及穿透依赖上游服务。
- 新部署流程已在本地 Linux 验证；完整新机 GPU 部署及客户端远程连接尚未实测。
- 提供简明使用说明、源码编译脚本、iKataGo 安装及恢复工具。
- 附 Linux 环境安装脚本，已在 Ubuntu 22.04 / RTX 5090 上完成安装、重复执行、完整编译和模型搜索验证。
- 运行需 SM120 GPU、适用驱动及 CUDA/cuDNN 运行库；不含正式模型。
- Windows 构建无 CUTLASS 融合 FFN，RTX 50 正向推理尚未实测。建议作为 Pre-release 发布。

详细测试范围见 VALIDATION.md；请用 SHA256SUMS.txt 校验附件。

相对原版的具体改动、启用条件及独立实验收益见 [CHANGES.md](CHANGES.md)。

Notebook 分别负责环境/引擎/默认最强评级权重、iKataGo 安装和文件整理、账号配置及启动。首次安装权重按官方 Strongest confidently-rated 标记选择。
