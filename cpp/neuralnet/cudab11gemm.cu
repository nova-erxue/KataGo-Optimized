#include "cudab11gemm.h"
#include <cstdint>
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"

namespace CudaB11Gemm {
namespace {
using T = cutlass::half_t;
using Layout = cutlass::layout::RowMajor;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
using Op = cutlass::epilogue::thread::LinearCombination<T,8,T,T>;
using Gemm = cutlass::gemm::device::Gemm<T,Layout,T,Layout,T,Layout,T,
  cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,
  cutlass::gemm::GemmShape<16,8,16>,Op,Swizzle,3>;
using Kernel = Gemm::GemmKernel;
}

cudaError_t prepare() {
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if(status != cudaSuccess) return status;
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties,device);
  if(status != cudaSuccess) return status;
  if(properties.major != 12 || properties.minor != 0) return cudaErrorNotSupported;
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(&attributes,cutlass::Kernel<Kernel>);
  if(status != cudaSuccess) return status;
  if(attributes.binaryVersion < 80) return cudaErrorNotSupported;
  return cudaFuncSetAttribute(cutlass::Kernel<Kernel>,cudaFuncAttributeMaxDynamicSharedMemorySize,
    sizeof(typename Kernel::SharedStorage));
}

cudaError_t run(const half* input,const half* weights,half* output,
  int outChannels,int tokens,int inChannels,bool accumulate,cudaStream_t stream) {
  if(!supportsShape(outChannels,tokens,inChannels) || !input || !weights || !output ||
     ((reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(weights) |
       reinterpret_cast<uintptr_t>(output)) & 15))
    return cudaErrorInvalidValue;
  const cutlass::gemm::GemmCoord problem(tokens,outChannels,inChannels);
  const auto tiled = Swizzle().get_tiled_shape(problem,{128,128,32},1);
  Kernel::Params params(problem,tiled,{(T*)input,inChannels},{(T*)weights,outChannels},
    {(T*)output,outChannels},{(T*)output,outChannels},Op::Params(T(1.0f),T(accumulate ? 1.0f : 0.0f)));
  cutlass::Kernel<Kernel><<<Swizzle().get_grid_shape(tiled),Kernel::kThreadCount,
    sizeof(typename Kernel::SharedStorage),stream>>>(params);
  return cudaPeekAtLastError();
}
}
