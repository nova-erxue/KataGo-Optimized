# 许可证检查记录

检查日期：2026-10-09。检查对象为本仓库初始提交 `cb921fb6fc50a0e5fba2b64e658c8b557cd80db5`、`v1.18.2-sm120.1` 的三个发布压缩包，以及本次声明补充。

## 结论

在下列已核查范围内，没有发现需要停止分发的明确协议冲突。发现并补充了新 FFN 特化文件的 CUTLASS 来源及版权声明；这是一次有范围限制的工程检查，不能保证所有文件来源和所有使用方式都已获得授权。

KataGo 的主体许可证允许修改及再分发，要求保留版权和许可声明；第三方组件仍分别受其许可证约束。依据：[KataGo 官方 LICENSE](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/LICENSE)。GitHub 将组合许可证识别为 `Other` 本身不代表违规。

## 已核查项目

| 项目 | 检查结果 |
|---|---|
| KataGo 主体 | 初始仓库的 LICENSE、CONTRIBUTORS 与官方基准提交逐字一致（忽略换行）。三个压缩包均包含这两个文件。未删除上游版权及免责声明。 |
| 上游许可文件 | 通过官方基准 Git tree 核对 LICENSE、LICENSE.txt、LICENSE.MIT、COPYING、NOTICE、LICENSE_AND_AUTHORS 文件名，未发现本仓库遗漏对应文件。此项不等同于逐行审计全部代码。 |
| CUTLASS | 原有 NVIDIA BSD-3-Clause 声明保留，源码和二进制包均包含 licenses/cutlass/LICENSE.txt。新 cudaffnregisterepilogue.cuh 的执行结构适配 DualGemm，本次补上完整原版权头、来源及修改说明，并在根 LICENSE 中明确该例外。没有打包受单独 EULA 约束的 python/CuTeDSL 目录。 |
| cuDNN frontend、其他外部源码 | 对 cpp/external 内许可文件与集中 licenses/ 副本进行核对，已有对应副本内容一致。原 CoreML NOTICE 已在源码中，本次额外复制到集中许可目录，方便再分发。CUDA 二进制不使用 CoreML。 |
| SHA2、Python 组件 | SHA2 源文件内保留其声明，并有 licenses/sha2.txt；Python 训练代码的 LICENSE_AND_AUTHORS 和 muon/LICENSE 保留。 |
| Linux libzip | Linux 附件中唯一单独打包的共享库为 lib/libzip.so.4，同时附带 Ubuntu 包的 licenses/libzip-copyright，包含 libzip 及内嵌代码的版权和条件。该文件也提及 Debian 打包文件的 GPL 条款，不能据此把 libzip 库本身一概判为 GPL。 |
| NVIDIA SDK | 压缩包没有单独附带 CUDA/cuDNN/TensorRT 动态库，含 NVIDIA-CUDA-EULA.txt。编译产生的静态运行时代码、用户另外安装的 SDK 仍受相应 NVIDIA 条款约束；本次未对二进制做完整链接成分重建。 |
| 模型 | 二进制包不含模型；源码包中的 10 个 .bin.gz 测试文件与官方基准提交的 Git blob 哈希逐一相同。没有打包正式 b11tf 权重。下载模型不意味着模型统一受引擎许可证覆盖，使用或再次分发模型须遵守其来源条款。 |
| iKataGo | 压缩包没有 iKataGo Server、FRP 程序或上游账号配置。安装时由用户机器下载上游包；项目已有 third_party/ikatago/NOTICE.md。下载方式减少本项目的副本分发范围，但不产生对 iKataGo 的修改、使用或再分发授权；本次未确认其完整授权范围。 |

## 发布附件核对

本次检查的本地文件 SHA256 与 GitHub Release API 公布的 digest 一致：

| 附件 | SHA256 |
|---|---|
| katago-v1.18.2-sm120.1-source.zip | `7b014db0783023dc1b9d23d00863e11e7d4e0efeab585be632d974bd705397e9` |
| katago-v1.18.2-sm120.1-linux-cuda13.2.tar.gz | `8cd2763659f880b30d95261f9a53ccfcf1801cfa9e09083b4f10d1cd50830d2c` |
| katago-v1.18.2-sm120.1-windows-cuda13.2.zip | `a96e2b5c7117691e836b8821af05c252584bf7253df4ad061f6bf83bc2970bdf` |

本次只修改声明和文档，没有修改计算逻辑或重新编译。已发布压缩包和 tag 保持原样，Release 增补 `THIRD_PARTY_NOTICES.txt`；再次分发这些旧包时请一并提供该补充文件。main 中的 FFN 文件头及集中 NOTICE 已补齐。

## 后续维护

- 保留上游和第三方版权、许可证、NOTICE，不用本项目的名称覆盖原作者署名。
- 新增依赖、SDK 二进制、正式模型或 iKataGo 文件时，应重新检查具体分发条款。
- 提交 KataGo 上游时保留第三方声明，拆分可审查的性能补丁；部署脚本和 SM120 发布限制不必一并提交。
- 本记录不覆盖商标、专利完整检索，也不能代替权利人对来源不明材料的授权。
