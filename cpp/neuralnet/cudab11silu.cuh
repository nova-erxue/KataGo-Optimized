#ifndef NEURALNET_CUDAB11SILU_CUH_
#define NEURALNET_CUDAB11SILU_CUH_
namespace B11Silu {
// Preserve the scalar path's half FMA, then FP32 exp/divide and half rounding.
template<int Channels> __global__ void affineSiluKernel(
  const half2* input, half2* output, const half2* scale, const half2* bias, int pairs
) {
#if __CUDA_ARCH__ >= 530
  const int i = blockIdx.x * 256 + threadIdx.x;
  if(i >= pairs) return;
  const int channel = i % (Channels / 2);
  const half2 affine = __hfma2(input[i],scale[channel],bias[channel]);
  const float2 f = __half22float2(affine);
  output[i] = __floats2half2_rn(f.x / (1.0f + expf(-f.x)),f.y / (1.0f + expf(-f.y)));
#endif
}
}
void customCudaB11Silu(const half* input,half* output,const half* scale,const half* bias,
                       int batch,int channels,cudaStream_t stream) {
  checkBufferIndexFitsInt(batch,361,channels,"customCudaB11Silu");
  const int pairs = batch * 361 * (channels / 2);
  if(channels == 384)
    B11Silu::affineSiluKernel<384><<<(pairs+255)/256,256,0,stream>>>(
      (const half2*)input,(half2*)output,(const half2*)scale,(const half2*)bias,pairs);
  else if(channels == 768)
    B11Silu::affineSiluKernel<768><<<(pairs+255)/256,256,0,stream>>>(
      (const half2*)input,(half2*)output,(const half2*)scale,(const half2*)bias,pairs);
  else throw std::runtime_error("customCudaB11Silu: unsupported channels");
}
#endif
