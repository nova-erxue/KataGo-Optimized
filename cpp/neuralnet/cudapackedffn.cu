#include "cudapackedffn.h"

#include <cstdint>
#include <limits>
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/activation.h"
#include "thread/left_silu_and_mul.h"

namespace CudaPackedFFN {
namespace {
using T = cutlass::half_t;
using Op = cutlass::epilogue::thread::LinearCombination<
  T, 8, T, T, cutlass::epilogue::thread::ScaleType::Nothing>;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;

// CUTLASS's regular epilogue reorders the accumulator into consecutive groups
// of eight half values. Packing makes the first/last four the two projections.
// Compute upstream SwiGLU while these values are still in registers and store
// only the four combined values. The ordinary GEMM never writes [M,2H].
template<class Base>
class SwiGLUOutput : public Base {
  typename Base::Element* output_;
  int rows_;
  int stride_;
public:
  using Element = typename Base::Element;
  using Fragment = typename Base::Fragment;
  using ThreadMap = typename Base::ThreadMap;
  using TensorCoord = typename Base::TensorCoord;
  using Params = typename Base::Params;
  static_assert(Base::kElementsPerAccess == 8, "packing requires eight-half accesses");

  CUTLASS_DEVICE SwiGLUOutput(
    Params const& params, Element* pointer, TensorCoord extent, int threadIdx,
    TensorCoord tileOffset = TensorCoord(), int const* indices = nullptr)
    : Base(params, pointer, extent, threadIdx, tileOffset, indices),
      output_(pointer), rows_(extent.row()), stride_(extent.column()/2) {}

  CUTLASS_HOST_DEVICE void add_pointer_offset(typename Base::LongIndex pointerOffset) {
    // Unused by the selected nonbatched GEMM, but keep the inherited iterator
    // API coherent: logical output offsets cover twice as many elements.
    assert(pointerOffset % 8 == 0);
    output_ += pointerOffset/2;
    Base::add_pointer_offset(pointerOffset);
  }

  CUTLASS_DEVICE void store_with_byte_offset(Fragment const& frag, int64_t byteOffset) const {
    // The selected nonbatched GEMM epilogue always passes zero. Keep the
    // iterator's byte-offset convention for completeness (the offset describes
    // the logical, twice-as-wide tensor before compression).
    Element* destination = reinterpret_cast<Element*>(
      reinterpret_cast<uint8_t*>(output_) + byteOffset/2);
    assert(byteOffset % 16 == 0);
    typename Base::Mask mask;
    Base::get_mask(mask);
    cutlass::epilogue::thread::LeftSiLUAndMul<Element,4,Element,Element> activation(
      typename cutlass::epilogue::thread::LeftSiLUAndMul<Element,4,Element,Element>::Params{});
    CUTLASS_PRAGMA_UNROLL
    for(int cluster=0; cluster<ThreadMap::Iterations::kCluster; cluster++) {
      CUTLASS_PRAGMA_UNROLL
      for(int group=0; group<ThreadMap::Iterations::kGroup; group++) {
        CUTLASS_PRAGMA_UNROLL
        for(int r=0; r<ThreadMap::Iterations::kRow; r++) {
          int fi = r + ThreadMap::Iterations::kRow*(group+ThreadMap::Iterations::kGroup*cluster);
          int row = Base::thread_start_row()+r*ThreadMap::Delta::kRow
            +group*ThreadMap::Delta::kGroup+cluster*ThreadMap::Delta::kCluster;
          CUTLASS_PRAGMA_UNROLL
          for(int c=0; c<ThreadMap::Iterations::kColumn; c++) {
            int col = Base::thread_start_column()+c*ThreadMap::Delta::kColumn;
            int offset = (fi*ThreadMap::Iterations::kColumn+c)*8;
            cutlass::Array<Element,4> lhs, rhs;
            CUTLASS_PRAGMA_UNROLL
            for(int j=0; j<4; j++) {
              lhs[j]=frag[offset+j];
              rhs[j]=frag[offset+4+j];
            }
            auto result = activation(lhs,rhs);
            cutlass::arch::global_store<decltype(result),sizeof(result)>(
              result, destination+int64_t(row)*stride_+col/2,
              row < rows_ && mask.predicates[c]);
          }
        }
      }
    }
  }
  CUTLASS_DEVICE void store(Fragment const& fragment) const {
    store_with_byte_offset(fragment,0);
  }
};

template<int M,int N,int WM,int WN,int Stages=3>
struct Types {
  using Base = typename cutlass::gemm::device::Gemm<
    T,cutlass::layout::RowMajor,T,cutlass::layout::ColumnMajor,
    T,cutlass::layout::RowMajor,T,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<M,N,32>,cutlass::gemm::GemmShape<WM,WN,32>,
    cutlass::gemm::GemmShape<16,8,16>,Op,Swizzle,Stages>::GemmKernel;
  using E = typename Base::Epilogue;
  using Epilogue = cutlass::epilogue::threadblock::Epilogue<
    typename E::Shape,typename E::WarpMmaOperator,E::kPartitionsK,
    SwiGLUOutput<typename E::OutputTileIterator>,typename E::AccumulatorFragmentIterator,
    typename E::WarpTileIterator,typename E::SharedLoadIterator,typename E::OutputOp,
    typename E::Padding>;
  using Kernel = cutlass::gemm::kernel::Gemm<typename Base::Mma,Epilogue,Swizzle,false>;
};

template<class TypesT>
cudaError_t prepareVariant() {
  using K = typename TypesT::Kernel;
  cudaFuncAttributes attr{};
  cudaError_t status=cudaFuncGetAttributes(&attr,cutlass::Kernel<K>);
  if(status!=cudaSuccess) return status;
  if(attr.binaryVersion<80) return cudaErrorNotSupported;
  return cudaFuncSetAttribute(cutlass::Kernel<K>,cudaFuncAttributeMaxDynamicSharedMemorySize,
                             sizeof(typename K::SharedStorage));
}

template<class TypesT>
cudaError_t launch(const half* a,const half* packed,half* output,int rows,int hidden,int channels,
                   cudaStream_t stream) {
  using K = typename TypesT::Kernel;
  using Shape = typename K::Mma::Shape;
  auto tiled=Swizzle().get_tiled_shape({rows,2*hidden,channels},
                                     {Shape::kM,Shape::kN,Shape::kK},1);
  // The logical output layout is [M,2H]. SwiGLUOutput compresses its stores to
  // [M,H]. Source C is never read because ScaleType::Nothing is used.
  typename K::Params params({rows,2*hidden,channels},tiled,
    {(T*)a,channels},{(T*)packed,channels},{nullptr,2*hidden},{(T*)output,2*hidden});
  cutlass::Kernel<K><<<Swizzle().get_grid_shape(tiled),K::kThreadCount,
                        sizeof(typename K::SharedStorage),stream>>>(params);
  return cudaPeekAtLastError();
}

// Packing preserves weight bits. Contiguous destination writes cover K first;
// each source column remains contiguous, exactly as in the original path.
__global__ void packKernel(const half* w1,const half* gate,half* packed,int hidden,int channels) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<2*hidden*channels) {
    int column=i/channels;
    int k=i%channels;
    int h=(column/8)*4+column%4;
    packed[i]=(column%8<4?w1:gate)[h*channels+k];
  }
}

#define DISPATCH(F) \
  switch(variant) { \
    case Variant::M64N128: return F(Types<64,128,32,64>); \
    case Variant::M128N128: return F(Types<128,128,64,64>); \
    case Variant::M128N256: return F(Types<128,256,64,64>); \
    case Variant::M64N256: return F(Types<64,256,32,64>); \
    case Variant::M256N128: return F(Types<256,128,64,64>); \
    case Variant::M128N128S4: return F(Types<128,128,64,64,4>); \
    default: return cudaErrorInvalidValue; \
  }
}

const char* name(Variant variant) {
  switch(variant) {
    case Variant::M64N128: return "64x128-s3";
    case Variant::M128N128: return "128x128-s3";
    case Variant::M128N256: return "128x256-s3";
    case Variant::M64N256: return "64x256-s3";
    case Variant::M256N128: return "256x128-s3";
    case Variant::M128N128S4: return "128x128-s4";
    default: return "invalid";
  }
}
cudaError_t prepare(Variant variant) {
  int device=0,major=0,minor=0;
  cudaError_t status=cudaGetDevice(&device);
  if(status!=cudaSuccess) return status;
  status=cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device);
  if(status!=cudaSuccess) return status;
  status=cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device);
  if(status!=cudaSuccess) return status;
  if(major!=12 || minor!=0) return cudaErrorNotSupported;
#define PREPARE(...) prepareVariant<__VA_ARGS__>()
  DISPATCH(PREPARE)
#undef PREPARE
}
cudaError_t packWeights(const half* w1,const half* gate,half* packed,int hidden,int channels,
                       cudaStream_t stream) {
  if(!w1 || !gate || !packed || hidden!=1152 || channels!=384 ||
     ((uintptr_t(w1)|uintptr_t(gate)|uintptr_t(packed))&15)) return cudaErrorInvalidValue;
  packKernel<<<(2*hidden*channels+255)/256,256,0,stream>>>(w1,gate,packed,hidden,channels);
  return cudaPeekAtLastError();
}
cudaError_t run(const half* a,const half* packed,half* output,int rows,int hidden,int channels,
               Variant variant,cudaStream_t stream) {
  if(!a || !packed || !output || rows<=0 || hidden!=1152 || channels!=384 ||
     rows>std::numeric_limits<int>::max()/(2*hidden) ||
     ((uintptr_t(a)|uintptr_t(packed)|uintptr_t(output))&15)) return cudaErrorInvalidValue;
#define LAUNCH(...) launch<__VA_ARGS__>(a,packed,output,rows,hidden,channels,stream)
  DISPATCH(LAUNCH)
#undef LAUNCH
}
#undef DISPATCH
}
