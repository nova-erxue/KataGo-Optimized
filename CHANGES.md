# 优化版相对原版改了什么

本版基于 KataGo 1.18.2，主要改动集中在 **CUDA 神经网络推理后端**，目的是减少 GPU 提交开销、重复读写和固定结构算子的执行时间。原始 TF/NBT 权重可以直接加载；模型结构、权重、网络计算量和 FP16 精度等级保持不变。

这里的“原版”是性能对照使用的官方 CUDA 实现，固定到 commit `fd0723fdbc0e9d82cf269c9630af8c27c57c07c4`，不表示与未来所有官方版本比较。此前也核对过官方 `d91ea855110dae533f0aada947b2b7d78cc8a4e1`，三份核心 CUDA / Attention / FFN 文件与该基准一致。

原版已经有合并 QKV 投影、MMA Flash-style Attention、CUTLASS 融合 FFN、残差与归一化融合、异步传输和临时缓冲复用。这些是本版继续使用的基础，不计为新增优化。

**实际保留的主要改动：**

| 改动 | 本版具体做法 | 启用条件或限制 |
|---|---|---|
| CUDA Graph | 按实际 batch 和执行分支缓存推理计算图，预热后用一次 Graph 提交网络，减少 CPU 逐个启动 kernel 的开销。每个 NN handle 独立管理；不能捕获时回到普通执行。 | `cudaUseGraphs=true`；发布配置不把 CPU↔GPU 传输纳入图。Graph 不会自动把多个 kernel 合成一个。 |
| GEMM 自动选择 | 在预热阶段比较原 cuBLAS 路径、cuBLASLt 候选及适用的固定形状 CUTLASS GEMM。候选需通过预热输入的逐位检查和配对计时，快于参考约 2% 才采用。正常推理不进行这类调参和临时分配。 | `cudaUseCublasLt=true`、`cudaB11Gemm=true`；保持原计算类型，不用 split-K。算法选择只缓存于当前进程，不写永久调优缓存。 |
| QKV 投影 + learned RoPE | 在投影 GEMM 输出阶段直接做旋转位置编码，减少单独 RoPE kernel 和中间张量读写。投影仍先按原语义舍入到 FP16，再做原 FP32 旋转。 | 固定 19 路、内层 C384、12×32 等匹配形状；小 batch 路径上限为 4，宽 tile 路径用于 B8/B16/B32。其他情况保留原投影和独立 RoPE。 |
| 独立 RoPE 向量化 | 没走投影融合时，为符合条件的布局提供向量读取和旋转实现。 | 发布配置 `cudaRopeVectorThreads=256`；由形状检查选择。 |
| 19 路专用 Attention | 针对 361 个位置、12 个头、每头 32 维、QKV stride 1152，简化通用索引，调整共享内存布局。K/V 缓冲由 20,480 减至 16,384 字节。 | `cudaB11Attention=true`；仅 SM120、FP16、无 mask 且布局/对齐符合条件。QK/PV 累加和 online softmax 顺序沿用原实现。 |
| B1 打包 FFN | 加载时在内存中交错打包两个投影矩阵，用单个 GEMM 加 SwiGLU 输出处理直接写出合并后的激活；不生成完整的双倍宽中间输出。 | `cudaB11PackedFFN=true`；仅实际 batch=1、19 路、C384/H1152。原始权重文件不改，大 batch 使用其他 FFN 路径。 |
| FFN 寄存器输出融合 | 保留 DualMma 主循环，在寄存器中完成相同的 SwiGLU 运算，再做一次输出重排和写回，减少输出阶段的共享内存处理。 | `cudaFusedFFNVariant=register-epilogue`；需 CUTLASS 和适用形状，保留原 FFN 回退。没有把整个 FFN 的两层 GEMM、归一化和残差全部融合。 |
| Compact RMSNorm | 对 C384/C768 重新安排每线程处理的数据量，减少线程数，同时保留原归约树、FP32 求和和残差舍入。 | `cudaCompactRMSNorm=true`；仅适用的 SM120 / FP16 Transformer 路径。 |
| SiLU 向量化 | 用 half2 成对处理仿射变换与 SiLU，保持 half FMA → FP32 exp/divide → half 的运算顺序。 | `cudaVectorSilu=true`；针对固定 19 路 C384/C768。 |

上述尺寸来自实际模型层描述。b11tf 名称中的 768 是外层宽度，不能认为每个内部 Transformer 都是 768 通道。其他 TF 权重仍可加载，但只有满足相应条件的层才使用专用路径；不能保证每份模型得到相同增益。

**调度、搜索和诊断方面：**

发布配置采用 48 搜索线程、最大 batch 16、2 个 NN server，这是 RTX 5090 + b11tf 的实测起点。源码中的 MCTS 搜索逻辑没有重写，也没有通过减少 visits、跳过网络层或改变输入特征来提速。最终版本没有加入新的自适应等待聚合调度器。

新增了固定 batch 原始输出对照、延迟采样，以及可选 NN batch 时间统计，方便定位问题和检查精度。正常运行不需要打开诊断。Graph 和 GEMM 调优在初始化/预热阶段准备资源，已观察的稳态推理窗口没有设备分配；这不等于承诺所有可能配置都完全无分配。

**实测收益应分开看：**

当前发布引擎在 RTX 5090、Linux、CUDA 13.2、cuDNN 9.24.1、b11tf s11003 上的同机单轮复核如下。NN 为 B16 / 1 NN server；搜索为 48 线程、V10000、12 个局面、B16 / 2 NN servers。

| 指标 | 原版 CUDA | 发布版 | 变化 |
|---|---:|---:|---:|
| NN eval/s | 3145.40 | 3512.87 | +11.68% |
| 整批平均延迟 | 5.0865 ms | 4.5546 ms | −10.46% |
| 搜索 visits/s | 6069.78 | 6568.29 | +8.21% |

下面是开发阶段的独立实验，用来说明改动的收益来源。各行的机器、batch 和对照范围不同，百分比不能相加，也不能直接当作搜索增幅：

| 实验 | 测量结果与范围 |
|---|---|
| 专用 Attention | B16 算子微基准耗时降低约 8.91%。 |
| B1 打包 FFN | 投影及 SwiGLU 微基准耗时降低约 34.03%。仅代表这个算子。 |
| 固定 GEMM / 向量 SiLU / 寄存器 FFN | B16 单轮整网筛选分别约 +2.21% / +1.09% / +0.98%；三项同时开启约 +3.58%。 |
| 宽 QKV+RoPE | 在其余优化已开启时，B8/B16/B32 三轮整网复测约 +1.21% / +0.70% / +1.19%。 |
| 寄存器 FFN 后续复测 | B16/B32/B64 约 +0.07% / +0.41% / +0.33%，独立贡献接近测量噪声，不是主要提速来源。 |

开发阶段 B16 trace 的 kernel 数量从每次推理 345 个降至 312 个，主要来自减少 33 个独立 RoPE kernel。CUDA Graph 的收益还包括减少 CPU 提交开销；不能把一次 Graph launch 理解成只有一个 GPU kernel。

精度验证中，5 份模型各使用同一组 669 个局面、固定 B16，与官方 FP16 原始输出逐位一致，原生精度检查也通过。覆盖 policy、value/score 和 ownership 等输出。此结论只适用于已测试输入，不是对任意模型和 batch 的逐位一致保证。测试条件见 [VALIDATION.md](VALIDATION.md)。

**试过但没有作为推荐路径的方案：**

- 整段 FFN 融合、Norm+QKV、大 batch 打包 FFN：没有稳定整网收益，撤回或不启用。
- 将 RoPE 融入 Attention 的 Q/K 加载：重复旋转 K，且影响异步加载，未保留。
- FFN 两级流水线、额外 tile/warp 变体、宽 QKV 四级流水线：没有优于保留实现的稳定收益。
- 输出头转换/池化融合、CPU ladder 特征候选：正确性验证后，未发现稳定收益，撤回。
- 包含传输的 CUDA Graph 和部分实验开关仍在源码中，发布配置默认关闭；不能把“源码有实现”当成“发布版启用”。

**发布和兼容性也做了调整：**

源码按计算能力检查 SM120，允许 RTX PRO 等产品，取消部署脚本对单个模型名字或 SHA 的绑定；权重文件可以按需指定。编译脚本支持 Windows/Linux，自动环境安装脚本目前支持 Ubuntu 22.04/24.04。Linux 包附带独立 libzip 和启动器，解决原版云镜像只替换裸引擎时缺库的问题。iKataGo 安装器沿用已有模型、合并优化参数并提供备份恢复。

**Linux 与 Windows 的优化范围不同。** Linux 发布版启用 CUTLASS。Windows 原生 MSVC 构建没有本版 CUTLASS 依赖的 QKV+RoPE 融合、固定 CUTLASS GEMM、打包 FFN 和寄存器 FFN 路径；仍保留可用的 Graph、cuBLASLt、专用 Attention、归一化和向量算子。Windows RTX 50 推理尚未实测，不能套用 Linux 的提速比例。

使用包内 `default.cfg` 或 iKataGo 安装器即可启用推荐组合。引擎新增的大部分开关在未配置时默认关闭，**只替换程序并沿用旧配置，不一定能获得上述提速**。关闭可选优化可回到本分支保留的通用 CUDA 路径，但 SM120 发布构建的架构限制仍然有效。

源码核对入口位于 `cpp/neuralnet/`：Graph 对应 `cudagraph.h`；GEMM 对应 `cudamatmultuner.h`、`cudab11gemm.cu`；RoPE 对应 `cudaropemm.cu`、`cudaropevector.cuh`；Attention 对应 `cudab11attention.cuh`；FFN 对应 `cudapackedffn.cu`、`cudaffnregisterepilogue.cuh`；Norm/SiLU 对应 `cudacompactnorm.cuh`、`cudab11silu.cuh`。条件选择和回退逻辑在 `cudaandrocmbackend.inc`。源码包包含这些文件，二进制包只附使用说明。
