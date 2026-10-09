#ifndef NEURALNET_CUDAROPEMM_H_
#define NEURALNET_CUDAROPEMM_H_
#include <cuda_fp16.h>
#include <cuda_runtime.h>
namespace CudaRopeMM {
struct Rope { const float* frequencies; int seqLen,boardX,queryHeads,kvHeads,headDim; };
// Fixed-shape QKV projection plus learnable RoPE. Uses the original in-major
// weights and rounds the projection to half before the unchanged FP32 rotation.
// The projection retains FP16 Tensor Core accumulation, without split-K.
cudaError_t prepare(int tileM);
cudaError_t run(const half* input,const half* weights,half* output,
  int rows,int columns,int channels,Rope rope,int tileM,cudaStream_t stream);
// SM120 wide QKV candidate: CTA 128x128x32, warp 64x64x32, three stages.
// Arithmetic is unchanged: FP16 accumulation and projection rounding, then the
// original FP32 rotation and half store. Call prepareWide once before capture;
// the backend selects the validated batch sizes and provides the SM120 guard.
cudaError_t prepareWide();
cudaError_t runWide(const half* input,const half* weights,half* output,
  int rows,int columns,int channels,Rope rope,cudaStream_t stream);
}
#endif
