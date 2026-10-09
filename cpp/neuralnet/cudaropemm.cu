#include "cudaropemm.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include <cstdint>
#include <type_traits>

namespace CudaRopeMM {
namespace {
template<class Base> class RotatingOutput : public Base {
  Rope rope_;
public:
  using Element=typename Base::Element;
  using Fragment=typename Base::Fragment;
  using ThreadMap=typename Base::ThreadMap;
  using TensorCoord=typename Base::TensorCoord;
  struct Params : Base::Params {
    Rope rope{};
    CUTLASS_HOST_DEVICE Params() {}
    CUTLASS_HOST_DEVICE Params(typename Base::Layout const& layout):Base::Params(layout){}
  };
  CUTLASS_DEVICE RotatingOutput(Params const& params,Element* pointer,TensorCoord extent,
    int threadIdx,TensorCoord tileOffset=TensorCoord(),int const* indices=nullptr)
    : Base(params,pointer,extent,threadIdx,tileOffset,indices),rope_(params.rope) {}
  CUTLASS_DEVICE void store_with_byte_offset(Fragment const& frag,int64_t offset) const {
    Fragment rotated=frag;
    if(rope_.frequencies) {
      CUTLASS_PRAGMA_UNROLL
      for(int cluster=0;cluster<ThreadMap::Iterations::kCluster;cluster++) {
        CUTLASS_PRAGMA_UNROLL
        for(int group=0;group<ThreadMap::Iterations::kGroup;group++) {
          CUTLASS_PRAGMA_UNROLL
          for(int r=0;r<ThreadMap::Iterations::kRow;r++) {
            int fi=r+ThreadMap::Iterations::kRow*(group+ThreadMap::Iterations::kGroup*cluster);
            int row=Base::thread_start_row()+r*ThreadMap::Delta::kRow+group*ThreadMap::Delta::kGroup+cluster*ThreadMap::Delta::kCluster;
            int xy=row%361;
            CUTLASS_PRAGMA_UNROLL
            for(int c=0;c<ThreadMap::Iterations::kColumn;c++) {
              int col=Base::thread_start_column()+c*ThreadMap::Delta::kColumn;
              int base=(fi*ThreadMap::Iterations::kColumn+c)*Base::kElementsPerAccess;
              CUTLASS_PRAGMA_UNROLL
              for(int pair=0;pair<Base::kElementsPerAccess;pair+=2) {
                int channel=col+pair;
                constexpr int qTotal=384;
                constexpr int kTotal=384;
                if(channel<qTotal+kTotal) {
                  int local=channel<qTotal?channel:channel-qTotal;
                  // b11tf has 12 Q/K heads of dimension 32. Each adjacent
                  // channel pair uses the same-index pair of XY frequencies.
                  int freq=local;
                  float angle=(float)(xy%19)*rope_.frequencies[freq]+(float)(xy/19)*rope_.frequencies[freq+1];
                  float sn,cs;__sincosf(angle,&sn,&cs);
                  float v0=float(frag[base+pair]),v1=float(frag[base+pair+1]);
                  rotated[base+pair]=Element(v0*cs-v1*sn);
                  rotated[base+pair+1]=Element(v0*sn+v1*cs);
                }
              }
            }
          }
        }
      }
    }
    Base::store_with_byte_offset(rotated,offset);
  }
  CUTLASS_DEVICE void store(Fragment const& frag) const { store_with_byte_offset(frag,0); }
};
using T=cutlass::half_t;
using Op=cutlass::epilogue::thread::LinearCombination<T,8,T,T,cutlass::epilogue::thread::ScaleType::Nothing>;
using Swizzle=cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
template<int M,int N=64> struct Types {
  // Preserve the old N64 epilogue. The wide candidate matches the measured
  // plain GEMM's alpha=1, beta=0 half epilogue before the same RoPE iterator.
  using ProjectionOp=std::conditional_t<N==64,Op,
    cutlass::epilogue::thread::LinearCombination<T,8,T,T>>;
  using Base=typename cutlass::gemm::device::Gemm<T,cutlass::layout::RowMajor,T,cutlass::layout::RowMajor,
    T,cutlass::layout::RowMajor,T,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<M,N,32>,cutlass::gemm::GemmShape<M/2,N/2,32>,cutlass::gemm::GemmShape<16,8,16>,ProjectionOp,Swizzle,3>::GemmKernel;
  using E=typename Base::Epilogue;
  using Epilogue=cutlass::epilogue::threadblock::Epilogue<typename E::Shape,typename E::WarpMmaOperator,
    E::kPartitionsK,RotatingOutput<typename E::OutputTileIterator>,typename E::AccumulatorFragmentIterator,
    typename E::WarpTileIterator,typename E::SharedLoadIterator,typename E::OutputOp,typename E::Padding>;
  using Kernel=cutlass::gemm::kernel::Gemm<typename Base::Mma,Epilogue,Swizzle,false>;
};
template<int M,int N=64> cudaError_t prepareKernel() {
  using K=typename Types<M,N>::Kernel;
  cudaFuncAttributes attr;cudaError_t e=cudaFuncGetAttributes(&attr,cutlass::Kernel<K>);
  if(e!=cudaSuccess)return e;
  if(attr.binaryVersion<80)return cudaErrorNotSupported;
  return cudaFuncSetAttribute(cutlass::Kernel<K>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(typename K::SharedStorage));
}
template<int M,int N=64> cudaError_t launch(const half* a,const half* b,half* out,int m,int n,int k,Rope rope,cudaStream_t stream) {
  using K=typename Types<M,N>::Kernel;
  auto tiled=Swizzle().get_tiled_shape({m,n,k},{M,N,32},1);
  typename K::Params params({m,n,k},tiled,{(T*)a,k},{(T*)b,n},{(T*)out,n},{(T*)out,n});
  params.params_D.rope=rope;
  cutlass::Kernel<K><<<Swizzle().get_grid_shape(tiled),K::kThreadCount,sizeof(typename K::SharedStorage),stream>>>(params);
  return cudaPeekAtLastError();
}
}
cudaError_t prepare(int tileM) {
  if(tileM==32)return prepareKernel<32>();
  if(tileM==64)return prepareKernel<64>();
  if(tileM==128)return prepareKernel<128>();
  return cudaErrorInvalidValue;
}
cudaError_t run(const half* input,const half* weights,half* output,int rows,int columns,int channels,Rope rope,int tileM,cudaStream_t stream) {
  if(!input || !weights || !output || !rope.frequencies || rows<=0 || columns<=0 || rows>2147483647/columns ||
    columns%64 || channels!=384 || columns!=1152 || rope.seqLen!=361 || rope.boardX!=19 ||
    rope.queryHeads!=12 || rope.kvHeads!=12 || rope.headDim!=32 ||
    rope.queryHeads<=0 || rope.kvHeads<=0 || rope.queryHeads%rope.kvHeads || rope.headDim<=0 || rope.headDim%8 ||
    (rope.queryHeads+2*rope.kvHeads)*rope.headDim!=columns ||
    ((uintptr_t(input)|uintptr_t(weights)|uintptr_t(output))&15))return cudaErrorInvalidValue;
  if(tileM==32)return launch<32>(input,weights,output,rows,columns,channels,rope,stream);
  if(tileM==64)return launch<64>(input,weights,output,rows,columns,channels,rope,stream);
  if(tileM==128)return launch<128>(input,weights,output,rows,columns,channels,rope,stream);
  return cudaErrorInvalidValue;
}
cudaError_t prepareWide() {
  return prepareKernel<128,128>();
}
cudaError_t runWide(const half* input,const half* weights,half* output,int rows,int columns,int channels,Rope rope,cudaStream_t stream) {
  if(!input || !weights || !output || !rope.frequencies || rows<=0 || columns<=0 || rows>2147483647/columns ||
    columns%64 || channels!=384 || columns!=1152 || rope.seqLen!=361 || rope.boardX!=19 ||
    rope.queryHeads!=12 || rope.kvHeads!=12 || rope.headDim!=32 ||
    rope.queryHeads<=0 || rope.kvHeads<=0 || rope.queryHeads%rope.kvHeads || rope.headDim<=0 || rope.headDim%8 ||
    (rope.queryHeads+2*rope.kvHeads)*rope.headDim!=columns ||
    ((uintptr_t(input)|uintptr_t(weights)|uintptr_t(output))&15))return cudaErrorInvalidValue;
  return launch<128,128>(input,weights,output,rows,columns,channels,rope,stream);
}
}
