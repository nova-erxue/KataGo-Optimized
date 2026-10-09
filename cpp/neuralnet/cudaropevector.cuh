// Fixed b11tf layout: [batch * 361, Q384 | K384 | V384].
// FP32 rotation and half rounding match the generic learnable RoPE kernel.
#ifndef NEURALNET_CUDAROPEVECTOR_CUH_
#define NEURALNET_CUDAROPEVECTOR_CUH_
template<int Threads>
__global__ void b11RopeVectorKernel(half2* qkv, const float2* frequencies, int pairs) {
  const int index = blockIdx.x * Threads + threadIdx.x;
  if(index >= pairs) return;
  const int row = index / 192;
  const int pair = index % 192;
  const int xy = row % 361;
  const float2 freq = frequencies[pair];
  const float angle = (float)(xy % 19) * freq.x + (float)(xy / 19) * freq.y;
  float sn,cs; __sincosf(angle,&sn,&cs);
  const int offset = row * 576 + pair;
  const float2 q = __half22float2(qkv[offset]);
  const float2 k = __half22float2(qkv[offset + 192]);
  qkv[offset] = __floats2half2_rn(q.x * cs - q.y * sn, q.x * sn + q.y * cs);
  qkv[offset + 192] = __floats2half2_rn(k.x * cs - k.y * sn, k.x * sn + k.y * cs);
}
void customCudaB11RopeVector(half* qkv,const float* frequencies,int batch,int threads,cudaStream_t stream) {
  checkBufferIndexFitsInt(batch,361,1152,"customCudaB11RopeVector");
  const int pairs=batch*361*192;
  if(threads==128)
    b11RopeVectorKernel<128><<<(pairs+127)/128,128,0,stream>>>((half2*)qkv,(const float2*)frequencies,pairs);
  else if(threads==256)
    b11RopeVectorKernel<256><<<(pairs+255)/256,256,0,stream>>>((half2*)qkv,(const float2*)frequencies,pairs);
  else
    throw std::runtime_error("customCudaB11RopeVector: threads must be 128 or 256");
}
#endif
