#ifndef NEURALNET_CUDACOMPACTNORM_CUH_
#define NEURALNET_CUDACOMPACTNORM_CUH_

// The C384/C768 kernels use 192/384 threads. Each original warp reduces
// 64 channels. Keep those reductions, but schedule three independent
// groups on each of two/four physical warps. Do not add the groups together in
// registers: that would change the FP32 reduction tree.
template<int CHANNELS, bool RESIDUAL>
__global__ void transformerCompactNormKernel(
  half* trunk, const half* delta, half* out, const half* gamma,
  const half* beta, const half* mask, float epsilon
) {
#ifdef KATAGO_GPU_SUPPORTS_FP16
  constexpr int THREADS = CHANNELS / 6;
  constexpr int WARPS = THREADS / 32;
  int pos = blockIdx.x;
  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  float maskVal = mask ? __half2float(mask[pos]) : 1.0f;
  half2* inRow = reinterpret_cast<half2*>(trunk + (size_t)pos * CHANNELS);
  const half2* deltaRow = RESIDUAL ? reinterpret_cast<const half2*>(delta + (size_t)pos * CHANNELS) : nullptr;
  half2* outRow = reinterpret_cast<half2*>(out + (size_t)pos * CHANNELS);
  float vals[6];
  __shared__ float warpSums[CHANNELS/64];
  #pragma unroll
  for(int group = 0; group < 3; group++) {
    int index = tid + group * THREADS;
    half2 x = inRow[index];
    if(RESIDUAL) {
      half2 d = deltaRow[index];
      half x0 = __float2half(__half2float(__low2half(x)) + __half2float(__low2half(d)) * maskVal);
      half x1 = __float2half(__half2float(__high2half(x)) + __half2float(__high2half(d)) * maskVal);
      x = __halves2half2(x0,x1);
      inRow[index] = x;
    }
    float v0 = __half2float(__low2half(x)) * maskVal;
    float v1 = __half2float(__high2half(x)) * maskVal;
    vals[2*group] = v0;
    vals[2*group+1] = v1;
    float acc = 0.0f;
    acc += v0 * v0 + v1 * v1;
    acc = warpReduceSumFloat(acc);
    if(lane == 0) warpSums[group*WARPS+warp] = acc;
  }
  __syncthreads();
  if(tid < 32) {
    float total = tid < CHANNELS/64 ? warpSums[tid] : 0.0f;
    total = warpReduceSumFloat(total);
    if(tid == 0) warpSums[0] = total;
  }
  __syncthreads();
  float rms = rsqrtf(warpSums[0] / (float)CHANNELS + epsilon);
  #pragma unroll
  for(int group = 0; group < 3; group++) {
    int index = tid + group*THREADS;
    half2 g = reinterpret_cast<const half2*>(gamma)[index];
    half2 b = reinterpret_cast<const half2*>(beta)[index];
    float y0 = vals[2*group] * rms * __half2float(__low2half(g)) + __half2float(__low2half(b));
    float y1 = vals[2*group+1] * rms * __half2float(__high2half(g)) + __half2float(__high2half(b));
    y0 *= maskVal;
    y1 *= maskVal;
    outRow[index] = __halves2half2(__float2half(y0),__float2half(y1));
  }
#else
  (void)trunk; (void)delta; (void)out; (void)gamma; (void)beta; (void)mask; (void)epsilon;
#endif
}

void customCudaTransformerCompactNorm(
  half* trunk, const half* delta, half* out, const half* gamma,
  const half* beta, const half* mask, int positions, int channels, float epsilon, cudaStream_t stream
) {
  if(positions <= 0) return;
  if(channels == 384) {
    if(delta) transformerCompactNormKernel<384,true><<<positions,64,0,stream>>>(trunk,delta,out,gamma,beta,mask,epsilon);
    else transformerCompactNormKernel<384,false><<<positions,64,0,stream>>>(trunk,delta,out,gamma,beta,mask,epsilon);
  } else if(channels == 768) {
    if(delta) transformerCompactNormKernel<768,true><<<positions,128,0,stream>>>(trunk,delta,out,gamma,beta,mask,epsilon);
    else transformerCompactNormKernel<768,false><<<positions,128,0,stream>>>(trunk,delta,out,gamma,beta,mask,epsilon);
  }
}
#endif
