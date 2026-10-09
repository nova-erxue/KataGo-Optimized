# KataGo 优化版引擎

基于 KataGo 1.18.2，面向 **所有 SM120（计算能力 12.0）GPU，包括 GeForce RTX 50 和 RTX PRO 6000 Blackwell** 的 CUDA 推理优化版。支持本版 KataGo 能读取的 TF/NBT 原始权重，无需转换模型；不修改权重、不做低精度量化。

这是第三方优化分支，不是 KataGo 官方发行版。发布附件采用 SM120 名称。安装脚本保留部分 `RTX50` 目录和配置名以兼容已有部署；引擎只按计算能力 12.0 检查。

想了解具体改了什么、哪些优化有效，以及 Windows/Linux 的差异，请看 [优化改动说明](CHANGES.md)。

项目采用统一名称，发布附件按 GPU 架构区分。当前提供 SM120 版本；30 系、40 系版本尚未发布，请勿使用当前 SM120 包。后续对应架构版本可放在同一仓库。

## 下载哪个文件

| 系统 | GitHub Release 附件 |
|---|---|
| Linux x86_64 | `katago-v1.18.2-sm120.1-linux-cuda13.2.tar.gz` |
| Windows x64 | `katago-v1.18.2-sm120.1-windows-cuda13.2.zip` |
| 自行编译 / 上传源码仓库 | `katago-v1.18.2-sm120.1-source.zip` |

需要 SM120 显卡、可运行 CUDA 13 的 NVIDIA 驱动和 CUDA/cuDNN 运行库。推荐 CUDA **13.2**、cuDNN **9.24.1**。包内不含正式模型和 NVIDIA 运行库。非 SM120 显卡会被引擎明确拒绝。

## Linux 云机：三个 Notebook 安装与启动（推荐）

需要 Ubuntu 22.04 / 24.04 x86_64、SM120 显卡、Python 3.10+ 和 Jupyter。安装环境需要 root 或免密 sudo；云主机沿用平台提供的 NVIDIA 驱动。

### 准备：上传并解压

将 Linux 发布包上传云机，在终端执行：

```bash
tar -xzf katago-v1.18.2-sm120.1-linux-cuda13.2.tar.gz
```

在 Jupyter 中进入解压得到的 `katago-sm120-linux` 文件夹。按下表顺序打开 Notebook，每份文件里的单元格从上到下运行，成功后再进入下一步。只上传 Notebook 文件不够，需要完整发布包。

| 顺序 | 文件 | 负责的工作 |
|---|---|---|
| ① | `katago.ipynb` | 安装 CUDA/cuDNN、优化引擎和默认权重 |
| ② | `ikatago install.ipynb` | 安装 iKataGo，把引擎、配置和权重放进对应目录 |
| ③ | `ikatago start.ipynb` | 设置账号密码并启动服务 |

### ① katago.ipynb：环境、引擎、权重

默认安装 CUDA **13.2**、cuDNN **9.24.x** 和本包优化引擎，从[官方模型页](https://katagotraining.org/networks/)下载 **Strongest confidently-rated network（最强且评级可信）**。模型下载后检查 gzip 完整性、记录 SHA256，并运行 GPU 加载和出招检查。

首次安装通常保留默认参数即可：

| 参数 | 默认值 / 用途 |
|---|---|
| `PACKAGE_DIR` | 当前工作目录，应为 Linux 发布包解压目录 |
| `ENGINE_DIR` | `~/katago-rtx50`，独立引擎安装目录 |
| `WORK_DIR` | `~/work`，下一步 iKataGo 安装目录 |
| `INSTALL_ENV` | `True`；环境已经装好时可改为 `False` |
| `LOCAL_MODEL` | 留空自动下载；也可以填写本地权重完整路径 |

“最强”按安装时的官方标记选择。2026-10-08 核对为 `kata1-tf3-b11c768-s11003M-d5973M-7gres`。重复运行会保留已安装权重；需要重新选择模型时使用新的 `ENGINE_DIR`。`ENGINE_DIR` 与 `WORK_DIR` 必须不同。

### ② ikatago install.ipynb：安装 iKataGo

自动读取第一步的路径，从上游下载安装 iKataGo Server 5.0.1，校验安装包 SHA256，再整理文件：

```text
~/work/
├── ikatago-server
├── data/bins/CUDA-Optimized-RTX50    # 优化引擎
├── data/weights/                   # 原始权重，保留文件名
├── data/configs/default_gtp.cfg     # 引擎配置
├── config/conf.yaml                # 自动生成的 iKataGo 配置
└── service.sh                      # 服务管理脚本
```

此步骤不设置账号、不启动服务。首次目标目录必须为空；重复运行会保留匹配的已有部署，不覆盖旧版 iKataGo。

发布包不附带 iKataGo 程序、穿透配置或平台令牌，它们在安装时从上游直接下载。上游包变化时会因校验不符停止安装。来源见 `third_party/ikatago/NOTICE.md`。

### ③ ikatago start.ipynb：账号与启动

首次按提示输入用户名和密码，密码通过隐藏输入框填写。启动后，在 [iKataGo 客户端](https://github.com/kinfkong/ikatago-client) 中使用相同账号连接；命令行客户端的平台参数为 `--platform all`。

- **日常启动：** 安装完成后，通常只需运行第三份 Notebook；已有账号会保留。
- **修改账号：** 先停服，再把 `RESET_ACCOUNT` 改为 `True`，运行账号设置单元格。
- **停止服务：** 把停止单元格的 `STOP_SERVICE` 改为 `True` 后运行；日常启动时保持 `False`。
- **关闭 Notebook：** 后台服务继续运行；关闭云实例会停止服务。

也可以在终端管理服务，自定义过 `WORK_DIR` 时替换路径：

```bash
~/work/service.sh start
~/work/service.sh status
~/work/service.sh stop
~/work/service.sh check   # 重新检查模型加载和出招
```

三个 Notebook 通过 `~/.config/katago-rtx50/notebook.json` 自动传递路径，请用同一个 Linux 用户运行。部署后的 `~/work` 可独立运行；通过 Notebook 启动还需要保留该路径记录。

### 常见问题

- **找不到发布包：** 将 `PACKAGE_DIR` 改为云机上的 Linux 发布包解压目录。
- **目标目录非空：** 修改 `WORK_DIR` 使用新目录；已有 iKataGo 按下文“接入已有 iKataGo”操作。
- **找不到 `config/conf.yaml`：** 全新安装运行第二份 Notebook，它会自动生成；`install_ikatago.py` 用于已有安装。
- **进程运行但客户端连不上：** 进程状态不代表远程注册成功。检查网络，并在云机查看 `~/work/logs/server.log`；远程发现和穿透依赖上游服务。

日志可能含账号信息，不要直接公开；分享 Notebook 前清空输出。三步流程已通过隔离测试，完整新机 GPU 部署及客户端远程连接仍待实测，见 [验证说明](VALIDATION.md)。

## 单独使用引擎：Linux 环境

Ubuntu 22.04 / 24.04 x86_64，在解压目录执行：

```bash
sudo bash setup-linux.sh
source /opt/katago-rtx50/env.sh
```

脚本安装 CUDA 13.2、cuDNN 9.24.x 和编译工具，完成后检查版本。云主机沿用宿主机驱动。只运行、不编译可加 `--runtime-only`；全新独立机器没有驱动时，按 [环境说明](ENVIRONMENT.md) 使用 `--install-driver`。root 用户省略 `sudo`。

## Linux：解压后运行

在解压目录执行，把 `model.bin.gz` 换成自己的权重路径：

```bash
chmod +x katago run-katago.sh
./run-katago.sh gtp -model /path/to/model.bin.gz -config default.cfg
```

棋盘软件的引擎路径选择 **`run-katago.sh`**，参数填 `gtp -model 权重路径 -config 配置路径`。

**原版云镜像实测：单独替换裸引擎会因缺少 `libzip.so.4` 启动失败。** Linux 包已带这个小型运行库，`run-katago.sh` 会自动加载，不需要安装系统包。保留整个解压目录，不能只复制启动脚本。这个库来自 Ubuntu 22.04，依赖系统的 OpenSSL 3、zlib 和 libbz2。

其他缺库问题用 `ldd katago` 排查。缺少 `libcublas.so.13`、`libnvrtc.so.13` 或 `libcudnn.so.9` 时，安装对应的 CUDA 13.2 / cuDNN 9.24.1 运行库；非默认目录可以这样启动：

```bash
KATAGO_LIBRARY_PATH=/path/to/cudnn/lib:/usr/local/cuda-13.2/lib64 ./run-katago.sh version
```

预编译程序需要 glibc ≥2.34、GLIBCXX_3.4.30 和前述运行库。遇到版本错误，按 [编译说明](BUILD.md) 在目标机器编译。

## Windows：解压后运行

1. 安装 CUDA 13.2、CUDA 13 对应的 cuDNN 9.24.1，以及 [Microsoft VC++ x64 运行库](https://aka.ms/vs/17/release/vc_redist.x64.exe)。
2. 将 cuDNN 的 **完整 DLL 集合**放进本包 `lib` 文件夹，或把它的 DLL 目录加入 `PATH`。CUDA 的 `bin` 目录也要在 `PATH` 中；`run-katago.cmd` 会优先使用 `CUDA_PATH_V13_2`。
3. 在解压目录打开终端：

```bat
run-katago.cmd version
run-katago.cmd gtp -model C:\models\model.bin.gz -config default.cfg
```

棋盘软件可选择 `katago.exe`，但启动该软件的环境也必须能找到上述 DLL。缺少 `cublas64_13.dll` / `nvrtc64_130_0.dll` 检查 CUDA；缺少 `cudnn64_9.dll` 检查 cuDNN。不要只下载一个 DLL。

Windows 已完成编译和启动验证，尚未在 Windows RTX 50 上实测推理。此 MSVC 构建未启用 CUTLASS 融合 FFN，不能套用 Linux 的性能提升。

## 不使用 Notebook：命令行部署 iKataGo

Linux 包包含 **优化引擎、libzip 和部署脚本**。iKataGo Server 5.0.1 和穿透配置在安装时从上游直接下载，发布包不附带 iKataGo 二进制或平台令牌。准备好原始 TF 权重，在解压目录执行：

```bash
bash deploy-ikatago.sh --setup-env --work /root/work --model /path/to/model.bin.gz
```

把权重路径换成自己的文件。脚本会安装 CUDA/cuDNN 环境，复制模型，生成配置并测试引擎，然后提示设置 iKataGo 用户名、密码。已有环境可去掉 `--setup-env`。目标目录必须为空，不会覆盖原来的 iKataGo。

```bash
/root/work/service.sh start    # 后台启动
/root/work/service.sh status   # 查看进程状态
/root/work/service.sh stop     # 停止
```

在 [iKataGo 客户端](https://github.com/kinfkong/ikatago-client) 中使用刚设置的用户名和密码连接；命令行客户端的平台参数为 `--platform all`。账号配置可停服后运行 `/root/work/service.sh configure` 修改。服务日志在 `/root/work/logs/server.log`，可能含账号信息，不要直接公开。

安装后 `/root/work` 可独立保留，不依赖解压目录。`service.sh check` 可重新检查模型加载及出招。此完整部署脚本仅支持 Linux x86_64 / Python 3.10+；Windows 包仍用于本地引擎。

联网安装环境、iKataGo 远程发现和穿透均需要网络。下载器校验固定 SHA256；上游包发生变化时会停止，需审查并更新脚本。公共平台凭据和远程服务由上游维护，不能保证永久可用，来源见 `third_party/ikatago/NOTICE.md`。包内不含个人账号、正式模型或 NVIDIA 运行库。

**验证范围：** 新部署文件生成和脚本流程已在本地 Linux 验证；当前云机连接已失效，完整的新机 GPU 部署和客户端远程连接尚未实测。此前引擎及环境脚本的云机验证见 [VALIDATION.md](VALIDATION.md)。

## 接入已有 iKataGo

此方法要求已有 `/root/work/config/conf.yaml`；没有这个文件时，按前面的三个 Notebook 安装。

先停止 iKataGo 服务，在发布包解压目录运行：

```bash
python3 install_ikatago.py --work /root/work
```

它会沿用当前默认权重，新增 `CUDA-Optimized-RTX50` 引擎和优化配置，保存备份，并在安装前后检查模型加载。原引擎仍可选。若提示缺少 PyYAML，运行 `python3 -m pip install PyYAML`。AutoDL 没有 `python3` 命令时可用 `/root/miniconda3/bin/python`。

安装完成后用生成的 **`/root/work/run-ikatago-rtx50.sh`** 启动原 iKataGo 服务，让服务继承正确的运行库路径。保留本发布包目录。Windows 把 `--work` 改成自己的目录，并使用生成的 `.cmd` 启动器。

需要恢复时：`python3 tools/restore_ikatago.py 安装时显示的备份目录`。

## 速度与配置

请使用包内 `default.cfg`，或通过安装脚本合并优化参数；只换程序、继续使用旧配置，很多优化不会开启。默认 **48 搜索线程 / batch 16 / 2 个 NN server** 是 5090 的起点，不是所有显卡的最优值。显存不足时先把 `nnMaxBatchSize` 改为 8。

```bash
./run-katago.sh benchmark -model /path/to/model.bin.gz -config default.cfg -visits 10000 -threads 40,48,64,80
```

此前 RTX 5090 / Linux / CUDA 13.2 / cuDNN 9.24.1 的同机单轮复核：NN eval/s **3145 → 3513**，B16 整批平均延迟 **5.086 → 4.555 ms**，48 线程 visits/s **6070 → 6568**。这不是所有模型或平台的保证。精度、测试条件和本次原版镜像兼容性记录见 [验证说明](VALIDATION.md)。

源码构建见 [BUILD.md](BUILD.md)。上游项目：[lightvector/KataGo](https://github.com/lightvector/KataGo)。许可证见 `LICENSE` 和 `licenses/`；源码保留各依赖的原始许可证。


许可证检查与第三方分发边界见 [LICENSE_AUDIT.md](LICENSE_AUDIT.md)。再次分发时请保留 `LICENSE`、`CONTRIBUTORS`、`licenses/` 和 `THIRD_PARTY_NOTICES.txt`。
