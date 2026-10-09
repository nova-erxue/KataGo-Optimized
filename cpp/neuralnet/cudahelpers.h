#ifndef NEURALNET_CUDAHELPERS_H_
#define NEURALNET_CUDAHELPERS_H_

#include "../neuralnet/cudaincludes.h"
#include "../neuralnet/activations.h"

#include "../neuralnet/cudaandrocmhelpers.h"

void customCudaTransformerCompactNorm(
  half* trunk, const half* delta, half* out, const half* gamma,
  const half* beta, const half* mask, int positions, int channels, float epsilon, cudaStream_t stream);

void customCudaB11RopeVector(half* qkv, const float* frequencies, int batch, int threads, cudaStream_t stream);

// SM120 fixed-shape BN/SiLU. Caller checks full-board, FP16 and alignment.
void customCudaB11Silu(const half* input, half* output, const half* scale,
                       const half* bias, int batch, int channels, cudaStream_t stream);

// Returns false for every unsupported device/shape/mask/alignment so the
// caller can use the original attention. Requires the per-handle MMA probe.
bool customCudaB11Attention(
  const half* q, const half* k, const half* v, const half* mask, half* out,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int vHeadDim,
  int qStride, int kvStride, int majorComputeCapability, int minorComputeCapability,
  cudaStream_t stream);

#endif  // NEURALNET_CUDAHELPERS_H_
