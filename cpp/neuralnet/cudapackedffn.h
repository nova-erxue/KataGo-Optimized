#ifndef KATAGO_EXPERIMENT_CUDAPACKEDFFN_H_
#define KATAGO_EXPERIMENT_CUDAPACKEDFFN_H_

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Experimental single-GEMM alternative to CudaFusedFFN. No runtime allocation,
// no intermediate projection tensors, no split K, and no precision changes.
namespace CudaPackedFFN {

enum class Variant {
  M64N128, M128N128, M128N256, M64N256, M256N128, M128N128S4,
  Count
};
const char* name(Variant variant);

// Initialize launch attributes once on the current device, before graph capture.
// SM120 only: an unsupported variant/device returns cudaErrorNotSupported.
cudaError_t prepare(Variant variant);

// Inputs are the existing FFN out-major weights [hidden, channels]. Output is
// [2*hidden, channels], with each 8-column GEMM group containing four W1 columns
// then the corresponding four Wgate columns. Pack once during model loading.
// Allocate exactly 2*hidden*channels half elements; all pointers are device pointers.
cudaError_t packWeights(
  const half* w1, const half* wGate, half* packed, int hidden, int channels,
  cudaStream_t stream = nullptr);

// A [rows,384]; packed [2304,384]; output [rows,1152], all contiguous.
// Output must not overlap inputs. All pointers require 16-byte alignment.
// SiLU and gate multiplication use the same half operations and rounding as
// the upstream DualGemm LeftSiLUAndMul epilogue. Test bit equality per variant.
cudaError_t run(
  const half* a, const half* packed, half* output,
  int rows, int hidden, int channels, Variant variant,
  cudaStream_t stream = nullptr);
}
#endif
