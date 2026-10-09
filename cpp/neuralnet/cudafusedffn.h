// Optional CUTLASS-based fused transformer FFN for the CUDA backend: one kernel computing
// SiLU(A @ W1) * (A @ Wgate) without writing the two intermediate GEMM outputs to global
// memory. Implemented in cudafusedffn.cu, which is compiled only when the build finds the
// vendored CUTLASS (CMake defines USE_CUTLASS_FUSED_FFN), so call sites must be guarded on
// that define.

#ifndef NEURALNET_CUDAFUSEDFFN_H_
#define NEURALNET_CUDAFUSEDFFN_H_

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <string>

namespace CudaFusedFFN {
  // Experimental choices are opt-in. All retain the original FP16 input, output and
  // accumulation types, K tile size and SwiGLU epilogue. Original is the upstream path.
  // PreparedOriginal launches the very same kernel after preparing its shared-memory
  // attribute once in supportedOnCurrentDevice(), rather than on every invocation.
  // RegisterEpilogue keeps the original GEMM mainloop and applies its pointwise
  // SwiGLU in registers before one shared-memory epilogue. SM120, N1152/K384;
  // other supported shapes use the prepared original DualGemm.
  enum class Variant { Original, PreparedOriginal, Tile64x64, Tile64x128, Tile128x128, RegisterEpilogue };
  Variant variantFromString(const std::string& name);
  const char* variantName(Variant variant);

  // Whether the fused FFN kernel actually works on the current device: runs a tiny dual GEMM
  // and verifies it executed, guarding against the same hazard as
  // flashAttentionMmaSupportedOnCurrentDevice in cudaflashmma.cuh (pre-sm_80 PTX JIT-compiling
  // to an empty stub) as well as any failure to launch with its large dynamic shared memory
  // requirement. Synchronous and slightly costly, so call once per handle creation.
  bool supportedOnCurrentDevice(Variant variant = Variant::Original);

  // Whether the kernel supports an FFN with weight matrices [N, K] (activations are [M, K]
  // with M varying per forward and not affecting support). Depends only on shape and
  // alignment, so callers may decide at model load time whether the fused path will be used
  // and commit to weight layouts accordingly.
  bool supportsShape(int N, int K, Variant variant = Variant::Original);

  // out = SiLU(A @ W1) * (A @ Wgate), FP16 io and FP16 accumulation (matching the unfused
  // cublasHgemm path). A is [M, K] row-major (NHWC tokens), w1/wGate are packed out-major
  // ([N, K] row-major), out is [M, N] row-major. All pointers must be 16-byte aligned.
  // The caller must have verified supportedOnCurrentDevice(variant) once on this device
  // and supportsShape(N, K, variant) beforehand. Keep variant unchanged for the handle's
  // lifetime, including graph capture/replay. Any failure here is a genuine error and
  // throws std::runtime_error; a launch error never silently retries another kernel.
  void runSwiGLU(
    const half* A, const half* w1, const half* wGate, half* out,
    int M, int N, int K, cudaStream_t stream, Variant variant = Variant::Original
  );
}

#endif  // NEURALNET_CUDAFUSEDFFN_H_
