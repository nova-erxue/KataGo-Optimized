# 从源码编译

源码不限模型文件和系统发行版；推理仅要求 GPU 计算能力为 12.0（SM120），不检查产品名称。Windows 和 Linux 必须分别编译。

准备 Python 3.8+、CMake、支持 CUDA 的 C++ 编译器、CUDA >=13.2、cuDNN >=9.24.0、zlib 开发文件。Linux 可用 GCC + Ninja；Windows 可用 Visual Studio 2022 C++ 工具链。脚本不会自动安装驱动或系统包。

在源码根目录运行。

## Linux

```bash
python3 tools/build.py --source . --build-dir build \
  --cuda-root /usr/local/cuda-13.2 --jobs 8
```

cuDNN 不在默认位置时，追加 `--cudnn-include /path/to/cudnn/include --cudnn-library /path/to/cudnn/lib/libcudnn.so`。

生成 `build/katago`。如果编译时找到 libzip，就会链接它；运行机器也需提供 libzip。编译机缺少 libzip 开发包时，GTP/分析仍可构建，但写训练数据功能不可用。

## Windows（PowerShell）

把示例中的 cuDNN 和 zlib 路径改成实际安装位置：

```powershell
python tools/build.py --source . --build-dir build `
  --generator "Visual Studio 17 2022" `
  --cuda-root "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2" `
  --cudnn-include C:/SDK/cudnn/include `
  --cudnn-library C:/SDK/cudnn/lib/x64/cudnn.lib `
  --cudnn-runtime C:/SDK/cudnn/bin/x64/cudnn64_9.dll `
  --zlib-include C:/SDK/zlib/include `
  --zlib-library C:/SDK/zlib/lib/zs.lib --jobs 6
```

生成 `build/Release/katago.exe`。zlib 的库名以实际构建结果为准。运行还需要完整 CUDA/cuDNN DLL 集合。

脚本使用 Release、CUDA backend、SM120 和 RTX 50 检查。`build-result.json` 记录工具链、程序 SHA256 与 CUTLASS 状态；不会通过一次编译宣称已验证 GPU 推理。当前 MSVC 路径不包含 CUTLASS 融合 FFN。

Linux 预编译版使用 CUDA 13.2.86 / cuDNN 9.24.1；Windows 使用 CUDA 13.2.51 / cuDNN 9.24.1 / MSVC 19.39。编译后的版本检查：`build/katago version` 或 `build/Release/katago.exe version`。

硬件限制由 `KATAGO_SM120_ONLY=ON` 控制，运行时只检查 major=12、minor=0。旧参数 `KATAGO_RTX50_ONLY` 保留为兼容入口，不再限制产品名。
