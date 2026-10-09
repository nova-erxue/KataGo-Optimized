#ifndef NEURALNET_CUDAB11GEMM_H_
#define NEURALNET_CUDAB11GEMM_H_

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <climits>

namespace CudaB11Gemm {
// Call once per compute handle, outside capture, after selecting its device.
// This experimental path is prepared only for SM120. Other devices keep cuBLAS.
cudaError_t prepare();

// Arguments use the existing cuBLAS dimensions: W[m,k] x X[k,n].
// The same buffers are X[n,k], W[k,m], C[n,m] in row-major order.
inline bool supportsShape(int m,int n,int k) {
  return n >= 361*8 && n % 361 == 0 && n <= INT_MAX / 1152 &&
    ((m == 1152 && k == 384) || (m == 384 && (k == 384 || k == 768 || k == 1152)));
}

// alpha is exactly 1, beta is 0/1, matching the backend's FP16 MatMul/1x1/QKV callers.
// No allocation, attribute setup, descriptors, or synchronization occur here.
cudaError_t run(const half* input,const half* weights,half* output,
  int outChannels,int tokens,int inChannels,bool accumulate,cudaStream_t stream);
}
#endif
