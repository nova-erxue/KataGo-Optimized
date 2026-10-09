// CUTLASS dual-GEMM + SwiGLU epilogue implementation of CudaFusedFFN (see cudafusedffn.h).
// Built on the DualGemm extension from CUTLASS's example 45 (vendored under
// external/cutlass/examples/45_dual_gemm, included via CMake): both GEMMs share the A operand
// tile pipeline, and the epilogue combines their accumulators as SiLU(D0) * D1 so the
// intermediates never touch global memory.
//
// The original tile configuration (threadblock 128x64x32, warp 64x32x32, 3 stages, FP16
// accumulation) remains the default: upstream measured it fastest on every architecture tested
// (sm_86 A5000, sm_89 4090, sm_90 H100, sm_120 RTX PRO 6000), 1.4-1.9x the unfused
// cublasHgemm x2 + SwiGLU kernel sequence at KataGo's FFN shapes. Additional tile choices
// are experimental; they have no performance or accuracy claim until separately measured.

#include "../neuralnet/cudafusedffn.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "device/dual_gemm.h"
#include "thread/left_silu_and_mul.h"
#include "cudaffnregisterepilogue.cuh"

namespace {

using ElementT = cutlass::half_t;
static constexpr int kAlignment = 128 / cutlass::sizeof_bits<ElementT>::value;

using EpilogueOutputOp01 = cutlass::epilogue::thread::LinearCombination<
  ElementT, kAlignment, ElementT, ElementT,
  cutlass::epilogue::thread::ScaleType::Nothing>;
using EpilogueOutputOp2 = cutlass::epilogue::thread::LeftSiLUAndMul<
  ElementT, kAlignment, ElementT, ElementT>;

template<int TileM, int TileN, int WarpM, int WarpN>
using DualGemmTile = cutlass::gemm::device::DualGemm<
  ElementT, cutlass::layout::RowMajor,
  ElementT, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor,
  ElementT, cutlass::layout::RowMajor,
  ElementT,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<TileM, TileN, 32>,
  cutlass::gemm::GemmShape<WarpM, WarpN, 32>,
  cutlass::gemm::GemmShape<16, 8, 16>,
  EpilogueOutputOp01, EpilogueOutputOp01, EpilogueOutputOp2,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
  3,      // stages
  false,  // kStoreD0
  false,  // kStoreD1
  false   // kSplitKSerial
>;

using DualGemm = DualGemmTile<128, 64, 64, 32>;
using RegisterEpilogueKernel = CudaFFNRegisterEpilogue::Kernel<DualGemm>;
using DualGemm64x64 = DualGemmTile<64, 64, 32, 32>;
using DualGemm64x128 = DualGemmTile<64, 128, 32, 64>;
using DualGemm128x128 = DualGemmTile<128, 128, 64, 64>;

template<typename Gemm>
typename Gemm::Arguments makeArgs(
  const half* A, const half* w1, const half* wGate, half* out, int M, int N, int K
) {
  return typename Gemm::Arguments(
    cutlass::gemm::DualGemmMode::kGemm,
    {M, N, K},
    {(ElementT const*)A, K},
    {(ElementT const*)w1, K},
    cutlass::TensorRef<ElementT const, cutlass::layout::RowMajor>(),
    cutlass::TensorRef<ElementT, cutlass::layout::RowMajor>(),
    {(ElementT const*)wGate, K},
    cutlass::TensorRef<ElementT const, cutlass::layout::RowMajor>(),
    cutlass::TensorRef<ElementT, cutlass::layout::RowMajor>(),
    {(ElementT*)out, N},
    EpilogueOutputOp01::Params(),
    EpilogueOutputOp01::Params(),
    EpilogueOutputOp2::Params(),
    1);
}

template<typename Gemm>
bool argsAreImplementable(const typename Gemm::Arguments& args) {
  if(Gemm::can_implement(args) != cutlass::Status::kSuccess)
    return false;
  // With split-K off no workspace is required. A nonzero requirement would mean the
  // configuration changed, so treat it as unsupported rather than allocating here.
  if(Gemm::get_workspace_size(args) != 0)
    return false;
  return true;
}

template<typename Gemm>
bool probeOnCurrentDevice() {
  // Never launch on a pre-sm_80 device: unlike KataGo's own arch-guarded kernels, CUTLASS's
  // below-arch fallback path is not an empty stub but a device-side trap
  // (CUTLASS_NOT_IMPLEMENTED), which would leave a sticky, unclearable error on the context.
  {
    int device = 0;
    int major = 0;
    if(cudaGetDevice(&device) != cudaSuccess)
      return false;
    if(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess)
      return false;
    if(major < 8)
      return false;
  }
  // Run a real tiny dual GEMM rather than a trivial probe kernel: this exercises the actual
  // kernel launch, including its 48KB+ dynamic shared memory opt-in, which a lighter probe
  // would not. With all-zero inputs the epilogue writes SiLU(0)*0 == 0, so pre-filling the
  // output with a nonzero pattern and checking it became zero verifies the kernel body truly
  // executed (a launch of an empty JIT stub would still report success).
  constexpr int M = 16;
  constexpr int N = 64;
  constexpr int K = 64;
  constexpr size_t numIn = (size_t)M * K + 2 * (size_t)N * K;
  constexpr size_t numOut = (size_t)M * N;
  half* buf = nullptr;
  if(cudaMalloc(&buf, (numIn + numOut) * sizeof(half)) != cudaSuccess)
    return false;
  half* A = buf;
  half* w1 = A + (size_t)M * K;
  half* wGate = w1 + (size_t)N * K;
  half* out = wGate + (size_t)N * K;

  bool ok = cudaMemset(buf, 0, numIn * sizeof(half)) == cudaSuccess;
  ok = ok && cudaMemset(out, 0xFF, numOut * sizeof(half)) == cudaSuccess;
  if(ok) {
    typename Gemm::Arguments args = makeArgs<Gemm>(A, w1, wGate, out, M, N, K);
    ok = argsAreImplementable<Gemm>(args);
    if(ok) {
      Gemm op;
      ok = op.initialize(args, nullptr, nullptr) == cutlass::Status::kSuccess;
      ok = ok && op.run(nullptr) == cutlass::Status::kSuccess;
    }
  }
  half hostOut[numOut];
  ok = ok && cudaMemcpy(hostOut, out, numOut * sizeof(half), cudaMemcpyDeviceToHost) == cudaSuccess;
  if(ok) {
    for(size_t i = 0; i < numOut; i++) {
      if(__half2float(hostOut[i]) != 0.0f) {
        ok = false;
        break;
      }
    }
  }
  // Clear any recoverable (non-sticky) launch error so an unsupported probe cannot leak error
  // state into later, unrelated CUDA calls. A device-side trap would be sticky and unclearable,
  // but the compute capability check above prevents the known trap path.
  (void)cudaGetLastError();
  (void)cudaFree(buf);
  return ok;
}

template<typename Gemm>
bool shapeIsSupported(int N, int K) {
  if(N <= 0 || K <= 0)
    return false;
  // can_implement depends on the dimensions and on operand alignment. M does not affect it
  // beyond positivity (A is row-major with lda == K, and the kernel predicates partial M
  // tiles), so a fixed representative M stands in for the runtime batch. The dummy pointers
  // carry the 16-byte alignment that the backend's device allocations guarantee.
  const int M = 128;
  const half* dummyA = reinterpret_cast<const half*>(uintptr_t(256));
  const half* dummyW = reinterpret_cast<const half*>(uintptr_t(256));
  half* dummyOut = reinterpret_cast<half*>(uintptr_t(256));
  return argsAreImplementable<Gemm>(makeArgs<Gemm>(dummyA, dummyW, dummyW, dummyOut, M, N, K));
}

// Equivalent to the vendored DualGemm::initialize()+run() for this wrapper's contract:
// kGemm, one split-K slice, no intermediate outputs and no workspace. Use the public
// kernel/Params API instead of reaching into the device wrapper's private params_.
// The probe already called Gemm::run() on this device, which sets the dynamic shared
// memory attribute. No global "prepared" flag is used: every handle probes its device.
template<typename Gemm, typename Kernel = typename Gemm::DualGemmKernel>
cutlass::Status runPrepared(const typename Gemm::Arguments& args, cudaStream_t stream) {
  static_assert(!Gemm::kSplitKSerial && !Gemm::kStoreD0 && !Gemm::kStoreD1,
                "prepared fused FFN requires no split-K and no intermediate output");
  typename Gemm::ThreadblockSwizzle swizzle;
  cutlass::gemm::GemmCoord tiledShape = swizzle.get_tiled_shape(
    args.problem_size,
    {Gemm::ThreadblockShape::kM, Gemm::ThreadblockShape::kN, Gemm::ThreadblockShape::kK},
    args.split_k_slices);
  typename Kernel::Params params{
    args.mode, args.problem_size, tiledShape,
    args.ref_A0.non_const_ref(), args.ref_B0.non_const_ref(),
    args.ref_C0.non_const_ref(), args.ref_D0,
    args.ref_B1.non_const_ref(), args.ref_C1.non_const_ref(), args.ref_D1, args.ref_D2,
    args.epilogue0, args.epilogue1, args.epilogue2,
    nullptr,
    args.batch_stride_A, args.batch_stride_B0, args.batch_stride_B1,
    args.batch_stride_C, args.batch_stride_D
  };
  dim3 grid = swizzle.get_grid_shape(tiledShape);
  dim3 block(Kernel::kThreadCount, 1, 1);
  cutlass::Kernel<Kernel><<<grid, block, sizeof(typename Kernel::SharedStorage), stream>>>(params);
  return cudaGetLastError() == cudaSuccess ? cutlass::Status::kSuccess : cutlass::Status::kErrorInternal;
}

template<typename Gemm, typename Kernel = typename Gemm::DualGemmKernel>
void runVariant(
  const half* A, const half* w1, const half* wGate, half* out,
  int M, int N, int K, cudaStream_t stream, bool prepared
) {
  // supportsShape probed can_implement with 16-byte-aligned dummy pointers, so enforce the same
  // alignment on the real operands rather than leaving it a documentation-only contract.
  if((uintptr_t(A) | uintptr_t(w1) | uintptr_t(wGate) | uintptr_t(out)) & 15)
    throw std::runtime_error("CudaFusedFFN::runSwiGLU: operand pointers must be 16-byte aligned");
  typename Gemm::Arguments args = makeArgs<Gemm>(A, w1, wGate, out, M, N, K);
  cutlass::Status status;
  if(prepared)
    status = runPrepared<Gemm, Kernel>(args, stream);
  else {
    Gemm op;
    status = op.initialize(args, nullptr, stream);
    if(status == cutlass::Status::kSuccess)
      status = op.run(stream);
  }
  if(status != cutlass::Status::kSuccess) {
    // The caller checked shape support at model load, so this is a genuine failure (a CUDA
    // error or an unexpected argument rejection), never a condition to silently fall back on.
    // Note run() ends with cudaGetLastError, so kErrorInternal may also reflect a pending
    // async error from an earlier kernel on this stream rather than this GEMM itself.
    throw std::runtime_error(
      std::string("CUTLASS fused FFN kernel failed (or a prior CUDA error was pending): ") +
      cutlass::cutlassGetStatusString(status) +
      ", M=" + std::to_string(M) + " N=" + std::to_string(N) + " K=" + std::to_string(K));
  }
}

// Used only by the new variant; preserve the original probe's behavior for
// existing variants. Unsupported kernels can fall back, but an allocation,
// execution or transfer failure must not be disguised as missing support.
bool checkRegisterProbeCuda(cudaError_t status, const char* operation) {
  if(status == cudaSuccess)
    return true;
  if(status == cudaErrorNotSupported || status == cudaErrorNoKernelImageForDevice ||
     status == cudaErrorInvalidDeviceFunction) {
    const cudaError_t pending = cudaGetLastError();
    if(pending != cudaSuccess && pending != cudaErrorNotSupported &&
       pending != cudaErrorNoKernelImageForDevice && pending != cudaErrorInvalidDeviceFunction)
      throw std::runtime_error(std::string("Register FFN probe ") + operation +
        ": " + cudaGetErrorString(status) + "; pending CUDA error: " + cudaGetErrorString(pending));
    return false;
  }
  throw std::runtime_error(std::string("Register FFN probe ") + operation + ": " + cudaGetErrorString(status));
}

template<typename Kernel>
bool probeRegisterVariantKernel() {
  if(!checkRegisterProbeCuda(cudaFuncSetAttribute(cutlass::Kernel<Kernel>,
       cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(typename Kernel::SharedStorage)),
       "set shared-memory attribute"))
    return false;
  constexpr int M = 16, N = 1152, K = 384;
  constexpr size_t numIn = (size_t)M * K + 2 * (size_t)N * K;
  constexpr size_t numOut = (size_t)M * N;
  std::vector<half> hostOut(numOut);
  struct DeviceBuffer {
    half* pointer = nullptr;
    // Cleanup must preserve the original exception, including a sticky CUDA
    // failure that can also make cudaFree fail.
    ~DeviceBuffer() { if(pointer) (void)cudaFree(pointer); }
  } buffer;
  if(!checkRegisterProbeCuda(cudaMalloc(&buffer.pointer, (numIn + numOut) * sizeof(half)), "allocate"))
    return false;
  half* a = buffer.pointer;
  half* w1 = a + (size_t)M * K;
  half* gate = w1 + (size_t)N * K;
  half* out = gate + (size_t)N * K;
  if(!checkRegisterProbeCuda(cudaMemset(buffer.pointer, 0, numIn * sizeof(half)), "clear input") ||
     !checkRegisterProbeCuda(cudaMemset(out, 0xFF, numOut * sizeof(half)), "fill output sentinel"))
    return false;
  typename DualGemm::ThreadblockSwizzle swizzle;
  auto tiledShape = swizzle.get_tiled_shape({M,N,K},{128,64,32},1);
  typename Kernel::Params params(cutlass::gemm::DualGemmMode::kGemm,{M,N,K},tiledShape,
    {(ElementT*)a,K},{(ElementT*)w1,K},{nullptr,N},{nullptr,N},
    {(ElementT*)gate,K},{nullptr,N},{nullptr,N},{(ElementT*)out,N});
  cutlass::Kernel<Kernel><<<swizzle.get_grid_shape(tiledShape),Kernel::kThreadCount,
    sizeof(typename Kernel::SharedStorage)>>>(params);
  // Inspect CUDA directly: runPrepared converts CUDA errors to a generic
  // CUTLASS status, which would lose the error detail required at initialization.
  if(!checkRegisterProbeCuda(cudaGetLastError(), "launch") ||
     !checkRegisterProbeCuda(cudaMemcpy(hostOut.data(), out, numOut * sizeof(half), cudaMemcpyDeviceToHost),
       "complete kernel and copy output"))
    return false;
  for(half value : hostOut) {
    if(__half2float(value) != 0.0f)
      throw std::runtime_error("Register FFN probe executed but failed the zero-output check");
  }
  return true;
}

bool probeRegisterEpilogueOnCurrentDevice() {
  int device = 0, major = 0, minor = 0;
  if(!checkRegisterProbeCuda(cudaGetDevice(&device), "get device") ||
     !checkRegisterProbeCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device), "get major capability") ||
     !checkRegisterProbeCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device), "get minor capability"))
    return false;
  if(major != 12 || minor != 0)
    return false;
  // Both the selected kernel and its other-shape fallback must be functional.
  // Avoid the legacy probe here because it intentionally folds CUDA errors into
  // false; the new variant's initialization must report real failures.
  return probeRegisterVariantKernel<typename DualGemm::DualGemmKernel>() &&
    probeRegisterVariantKernel<RegisterEpilogueKernel>();
}

}  // namespace

namespace CudaFusedFFN {

Variant variantFromString(const std::string& name) {
  if(name == "original") return Variant::Original;
  if(name == "original-prepared") return Variant::PreparedOriginal;
  if(name == "register-epilogue") return Variant::RegisterEpilogue;
  if(name == "64x64") return Variant::Tile64x64;
  if(name == "64x128") return Variant::Tile64x128;
  if(name == "128x128") return Variant::Tile128x128;
  throw std::invalid_argument("Unknown CUDA fused FFN variant: " + name);
}

const char* variantName(Variant variant) {
  switch(variant) {
    case Variant::Original: return "original";
    case Variant::PreparedOriginal: return "original-prepared";
    case Variant::RegisterEpilogue: return "register-epilogue";
    case Variant::Tile64x64: return "64x64";
    case Variant::Tile64x128: return "64x128";
    case Variant::Tile128x128: return "128x128";
  }
  throw std::invalid_argument("Unknown CUDA fused FFN variant enum");
}

bool supportedOnCurrentDevice(Variant variant) {
  switch(variant) {
    case Variant::RegisterEpilogue: return probeRegisterEpilogueOnCurrentDevice();
    case Variant::Original:
    case Variant::PreparedOriginal: return probeOnCurrentDevice<DualGemm>();
    case Variant::Tile64x64: return probeOnCurrentDevice<DualGemm64x64>();
    case Variant::Tile64x128: return probeOnCurrentDevice<DualGemm64x128>();
    case Variant::Tile128x128: return probeOnCurrentDevice<DualGemm128x128>();
  }
  return false;
}

bool supportsShape(int N, int K, Variant variant) {
  switch(variant) {
    case Variant::Original:
    case Variant::RegisterEpilogue:
    case Variant::PreparedOriginal: return shapeIsSupported<DualGemm>(N, K);
    case Variant::Tile64x64: return shapeIsSupported<DualGemm64x64>(N, K);
    case Variant::Tile64x128: return shapeIsSupported<DualGemm64x128>(N, K);
    case Variant::Tile128x128: return shapeIsSupported<DualGemm128x128>(N, K);
  }
  return false;
}

void runSwiGLU(
  const half* A, const half* w1, const half* wGate, half* out,
  int M, int N, int K, cudaStream_t stream, Variant variant
) {
  switch(variant) {
    case Variant::RegisterEpilogue:
      if(N == 1152 && K == 384)
        return runVariant<DualGemm, RegisterEpilogueKernel>(A, w1, wGate, out, M, N, K, stream, true);
      return runVariant<DualGemm>(A, w1, wGate, out, M, N, K, stream, true);
    case Variant::Original:
      return runVariant<DualGemm>(A, w1, wGate, out, M, N, K, stream, false);
    case Variant::PreparedOriginal:
      return runVariant<DualGemm>(A, w1, wGate, out, M, N, K, stream, true);
    case Variant::Tile64x64:
      return runVariant<DualGemm64x64>(A, w1, wGate, out, M, N, K, stream, true);
    case Variant::Tile64x128:
      return runVariant<DualGemm64x128>(A, w1, wGate, out, M, N, K, stream, true);
    case Variant::Tile128x128:
      return runVariant<DualGemm128x128>(A, w1, wGate, out, M, N, K, stream, true);
  }
  throw std::invalid_argument("Unknown CUDA fused FFN variant enum");
}

}  // namespace CudaFusedFFN
