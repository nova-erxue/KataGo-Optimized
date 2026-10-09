# 发布 KataGo 优化版引擎到 GitHub

建议仓库名 **KataGo-Optimized**，版本标签 **v1.18.2-sm120.1**，Release 标题 **KataGo 优化版引擎 — SM120 / CUDA 13.2**。

本版支持所有计算能力为 **12.0（SM120）** 的 GPU，包括 GeForce RTX 50 和 RTX PRO 6000 Blackwell。SM100、SM121 等其他架构不属于此预编译版的支持范围。

项目名称固定为“**KataGo 优化版引擎**”，仓库建议名为 `KataGo-Optimized`。SM120 只表示本次附件支持的架构，不是项目名称。后续 40 系、30 系版本继续使用同一个仓库，分别注明实际支持架构、CUDA 版本和测试范围。当前这些版本尚未发布。

## 1. 准备文件

本次 Release 上传这四个附件，不要混用旧 RTX50 包和旧校验文件：

```text
katago-v1.18.2-sm120.1-source.zip
katago-v1.18.2-sm120.1-linux-cuda13.2.tar.gz
katago-v1.18.2-sm120.1-windows-cuda13.2.zip
SHA256SUMS.txt
```

解压源码包得到 `KataGo-Optimized`。里面的内容上传到源码仓库；压缩包放在 Release 附件区。

## 2. 上传源码

在 GitHub 创建空仓库 `KataGo-Optimized`，不用自动生成 README 或许可证。在本地 `KataGo-Optimized` 源码目录打开 PowerShell，执行：

```powershell
git init
git add .
git commit -m "Release optimized KataGo SM120"
git branch -M main
git remote add origin https://github.com/你的用户名/KataGo-Optimized.git
git push -u origin main
```

将“你的用户名”替换为自己的 GitHub 用户名。已有仓库直接更新文件、提交和推送，不用重复初始化。需要改仓库名时，在 GitHub 的 Settings → General 修改 Repository name，并更新本地 origin 地址。

保留 LICENSE、CONTRIBUTORS 和 licenses/。不要上传云机账号配置、密码、私钥、模型、部署后的 work 目录或下载缓存。

## 3. 创建 Release

进入仓库页面的 **Releases → Draft a new release**，填写：

| 项目 | 内容 |
|---|---|
| Tag | `v1.18.2-sm120.1`，创建新标签 |
| Target | 包含本版源码的 `main` 提交 |
| Title | `KataGo 优化版引擎 — SM120 / CUDA 13.2` |
| Description | 复制本包 `RELEASE_NOTES.md` 的内容 |

把说明中 CHANGES.md 的相对链接换成自己的仓库文件链接：

```text
https://github.com/你的用户名/KataGo-Optimized/blob/v1.18.2-sm120.1/CHANGES.md
```

上传第一步的四个附件，等上传完成。勾选 **Set as a pre-release**：Windows 已重新编译并完成启动检查，但 Windows SM120 正向推理及完整 iKataGo 客户端连接尚未实测。然后点击 **Publish release**。

已有 RTX50 公开版本建议保留，另发这个新标签，避免旧附件与新源码混用。

## 4. 检查发布结果

确认 Release 页面有四个附件，Linux 包解压目录为 `katago-sm120-linux`，包含：

```text
katago.ipynb
ikatago install.ipynb
ikatago start.ipynb
```

用户下载 Linux 或 Windows 编译包即可运行。GitHub 自动生成的 Source code 附件只含源码。

iKataGo 在安装时由用户机器从上游下载，本发布包不附带服务端、穿透配置或平台令牌。Notebook 发布时保持输出为空。

本次已整理附件和教程，尚未创建远程仓库或上传 GitHub。
