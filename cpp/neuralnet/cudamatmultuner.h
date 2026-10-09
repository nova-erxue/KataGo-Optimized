#ifndef NEURALNET_CUDAMATMULTUNER_H_
#define NEURALNET_CUDAMATMULTUNER_H_

#include <cublasLt.h>
#include <map>
#include <memory>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include "../core/logger.h"
#ifdef USE_CUTLASS_FUSED_FFN
#include "cudab11gemm.h"
#endif

// One instance per compute handle. Descriptors/algorithms live as long as CUDA
// graphs that reference them; the owning handle synchronizes before teardown.
// All data, scale and compute types match the existing CUDA Hgemm path.
class CudaMatmulTuner {
public:
  enum class BaselineKind { Hgemm, StridedBatchedOne };

private:
  struct Plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo{};
    bool useLt = false;
    bool useFixed = false;
    ~Plan() {
      if(op) (void)cublasLtMatmulDescDestroy(op);
      if(a) (void)cublasLtMatrixLayoutDestroy(a);
      if(b) (void)cublasLtMatrixLayoutDestroy(b);
      if(c) (void)cublasLtMatrixLayoutDestroy(c);
    }
  };
  struct Temporary {
    void* output = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    ~Temporary() {
      if(output) (void)cudaFree(output);
      if(begin) (void)cudaEventDestroy(begin);
      if(end) (void)cudaEventDestroy(end);
      if(pref) (void)cublasLtMatmulPreferenceDestroy(pref);
    }
  };
  cublasLtHandle_t lt = nullptr;
  void* workspace = nullptr;
  static constexpr size_t workspaceBytes = 32 * 1024 * 1024;
  using Key = std::tuple<int,int,int,bool,BaselineKind,long long>;
  std::map<Key,std::unique_ptr<Plan>> plans;
  Logger* logger;
  bool fixedReady = false;

  static bool unavailable(cublasStatus_t s) {
    return s == CUBLAS_STATUS_NOT_SUPPORTED || s == CUBLAS_STATUS_ARCH_MISMATCH;
  }

  // These queries are diagnostics only. An unavailable diagnostic attribute must
  // not turn a successfully tested algorithm into an inference failure.
  template<typename T = uint32_t>
  static std::string configValue(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t attr) {
    T value = 0;
    size_t written = 0;
    cublasStatus_t status = cublasLtMatmulAlgoConfigGetAttribute(&algo,attr,&value,sizeof(value),&written);
    return status == CUBLAS_STATUS_SUCCESS && written == sizeof(value)
      ? std::to_string(value) : "unavailable";
  }

  static std::string algorithmDetails(const cublasLtMatmulAlgo_t& algo) {
    uint64_t numericalFlags = 0;
    size_t written = 0;
    cublasStatus_t status = cublasLtMatmulAlgoCapGetAttribute(&algo,
      CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,&numericalFlags,sizeof(numericalFlags),&written);
    return " lt_algo_id=" + configValue<int32_t>(algo,CUBLASLT_ALGO_CONFIG_ID) +
      " lt_tile=" + configValue(algo,CUBLASLT_ALGO_CONFIG_TILE_ID) +
      " lt_stages=" + configValue(algo,CUBLASLT_ALGO_CONFIG_STAGES_ID) +
      " lt_split_k=" + configValue(algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM) +
      " lt_reduction=" + configValue(algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME) +
      " lt_swizzle=" + configValue(algo,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING) +
      " lt_custom=" + configValue(algo,CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION) +
#if defined(CUBLAS_VER_MAJOR) && CUBLAS_VER_MAJOR >= 13
      " lt_inner_shape=" + configValue<uint16_t>(algo,CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID) +
      " lt_cluster_shape=" + configValue<uint16_t>(algo,CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID) +
#endif
      " lt_numerical_flags=" + (status == CUBLAS_STATUS_SUCCESS && written == sizeof(numericalFlags)
        ? std::to_string(numericalFlags) : "unavailable");
  }

  void tune(Plan& p, cublasHandle_t blas, cudaStream_t stream,
            int m, int n, int k, const void* a, const void* b, const void* c,
            const cublas_half_t* alpha, const cublas_half_t* beta, bool accumulate,
            BaselineKind baselineKind, long long outputStrideElts) {
    CUBLAS_ERR("Lt descriptor",cublasLtMatmulDescCreate(&p.op,CUBLAS_COMPUTE_16F,CUDA_R_16F));
    CUBLAS_ERR("Lt layout",cublasLtMatrixLayoutCreate(&p.a,CUDA_R_16F,m,k,m));
    CUBLAS_ERR("Lt layout",cublasLtMatrixLayoutCreate(&p.b,CUDA_R_16F,k,n,k));
    CUBLAS_ERR("Lt layout",cublasLtMatrixLayoutCreate(&p.c,CUDA_R_16F,m,n,m));
    Temporary t;
    CUBLAS_ERR("Lt preference",cublasLtMatmulPreferenceCreate(&t.pref));
    CUBLAS_ERR("Lt workspace",cublasLtMatmulPreferenceSetAttribute(t.pref,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspaceBytes,sizeof(workspaceBytes)));
    const uint32_t alignment = 16;
    for(auto attr : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
      CUBLAS_ERR("Lt alignment",cublasLtMatmulPreferenceSetAttribute(t.pref,attr,&alignment,sizeof(alignment)));
    cublasLtMatmulHeuristicResult_t candidates[16];
    int count = 0;
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(lt,p.op,p.a,p.b,p.c,p.c,t.pref,16,candidates,&count);
    if(unavailable(status)) return;
    CUBLAS_ERR("Lt heuristic",status);
    if(count == 0) return;
    const size_t bytes = (size_t)m * n * sizeof(cublas_half_t);
    CUDA_ERR("Lt tune buffer",cudaMalloc(&t.output,bytes));
    CUDA_ERR("Lt tune event",cudaEventCreate(&t.begin));
    CUDA_ERR("Lt tune event",cudaEventCreate(&t.end));
    // Restore the residual before EVERY probe, outside its measured interval.
    // Tuning never writes into any live model activation or weight buffer.
    auto measure = [&](const cublasLtMatmulAlgo_t* algorithm, float& millis) {
      float samples[5];
      for(int repeat = 0; repeat < 6; repeat++) {
        if(accumulate) CUDA_ERR("Lt restore residual",cudaMemcpyAsync(t.output,c,bytes,cudaMemcpyDeviceToDevice,stream));
        CUDA_ERR("Lt timer",cudaEventRecord(t.begin,stream));
        cublasStatus_t s;
        if(algorithm)
          s = cublasLtMatmul(lt,p.op,alpha,a,p.a,b,p.b,beta,t.output,p.c,t.output,p.c,
                            algorithm,workspace,workspaceBytes,stream);
        else if(baselineKind == BaselineKind::StridedBatchedOne)
          s = cublasHgemmStridedBatched(blas,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,alpha,
            (const cublas_half_t*)a,m,(long long)m*k,(const cublas_half_t*)b,k,0LL,
            beta,(cublas_half_t*)t.output,m,outputStrideElts,1);
        else
          s = cublasHgemm(blas,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,alpha,
            (const cublas_half_t*)a,m,(const cublas_half_t*)b,k,beta,(cublas_half_t*)t.output,m);
        if(unavailable(s)) return false;
        CUBLAS_ERR("Lt tune launch",s);
        CUDA_ERR("Lt timer",cudaEventRecord(t.end,stream));
        CUDA_ERR("Lt tune completion",cudaEventSynchronize(t.end));
        float elapsed;
        CUDA_ERR("Lt timer",cudaEventElapsedTime(&elapsed,t.begin,t.end));
        if(repeat > 0) samples[repeat-1] = elapsed;
      }
      std::sort(samples,samples+5);
      millis = samples[2];
      return std::isfinite(millis) && millis > 0.0f;
    };
    float baselineMs;
    if(!measure(nullptr,baselineMs)) return;
    float bestMs = baselineMs * 0.98f;
    for(int i = 0; i < count; i++) {
      if(candidates[i].state != CUBLAS_STATUS_SUCCESS || candidates[i].workspaceSize > workspaceBytes) continue;
      // Do not change the reduction into a split-K accumulation scheme.
      uint32_t splitK = 0;
      size_t written = 0;
      CUBLAS_ERR("Lt split K",cublasLtMatmulAlgoConfigGetAttribute(&candidates[i].algo,
        CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&splitK,sizeof(splitK),&written));
      if(splitK > 1) continue;
      float ms;
      if(measure(&candidates[i].algo,ms) && ms < bestMs) {
        bestMs = ms;
        p.algo = candidates[i].algo;
        p.useLt = true;
      }
    }

    const bool foundCandidate = p.useLt;
    std::string verification = "no-candidate";
    std::string verificationDetails;
    if(foundCandidate) {
      // Recheck the finalist against the actual fallback, alternating which one
      // runs first. Require the median paired ratio to beat it by 2% (at least
      // two of three pairs). Other streams and clock changes can still perturb
      // these measurements; this is not a substitute for end-to-end benchmarks.
      float baselineChecks[3], candidateChecks[3], ratios[3];
      bool complete = true;
      for(int pair = 0; pair < 3; pair++) {
        bool ok;
        if(pair % 2 == 0)
          ok = measure(nullptr,baselineChecks[pair]) && measure(&p.algo,candidateChecks[pair]);
        else
          ok = measure(&p.algo,candidateChecks[pair]) && measure(nullptr,baselineChecks[pair]);
        if(!ok) { complete = false; break; }
        ratios[pair] = candidateChecks[pair] / baselineChecks[pair];
      }
      if(complete) {
        std::sort(baselineChecks,baselineChecks+3);
        std::sort(candidateChecks,candidateChecks+3);
        std::sort(ratios,ratios+3);
        baselineMs = baselineChecks[1];
        bestMs = candidateChecks[1];
        p.useLt = ratios[1] < 0.98f;
        verification = p.useLt ? "pass" : "reject";
        verificationDetails = " paired_ratio_min=" + std::to_string(ratios[0]) +
          " paired_ratio_median=" + std::to_string(ratios[1]) +
          " paired_ratio_max=" + std::to_string(ratios[2]);
      }
      else {
        p.useLt = false;
        verification = "unavailable";
      }
    }
    std::string warmupAccuracy = "not-selected";
    if(p.useLt) {
      // A newly selected tactic must also reproduce the fallback's warmup
      // output bit for bit. This conservative guard is outside timed inference;
      // testing many real positions is still required because warmup uses one
      // board and the cache is shared by layers with the same matrix shape.
      std::vector<uint16_t> expected(bytes/sizeof(uint16_t)), actual(expected.size());
      float ignoredMs;
      bool ok = measure(nullptr,ignoredMs);
      if(ok) CUDA_ERR("Lt reference output",cudaMemcpyAsync(expected.data(),t.output,bytes,cudaMemcpyDeviceToHost,stream));
      if(ok) CUDA_ERR("Lt reference output",cudaStreamSynchronize(stream));
      ok = ok && measure(&p.algo,ignoredMs);
      if(ok) CUDA_ERR("Lt candidate output",cudaMemcpyAsync(actual.data(),t.output,bytes,cudaMemcpyDeviceToHost,stream));
      if(ok) CUDA_ERR("Lt candidate output",cudaStreamSynchronize(stream));
      p.useLt = ok && expected == actual;
      warmupAccuracy = p.useLt ? "exact" : "reject";
    }
    if(logger) logger->write("Cuda backend: GEMM tuning " + std::to_string(m) + "x" + std::to_string(n) + "x" +
      std::to_string(k) + " beta=" + std::to_string(accumulate ? 1 : 0) +
      " baseline=" + (baselineKind == BaselineKind::Hgemm ? "Hgemm" : "HgemmStridedBatchedOne") +
      " output_stride=" + std::to_string(outputStrideElts) + " selected " +
      (p.useLt ? "cuBLASLt" : "cuBLAS") + " original_ms=" + std::to_string(baselineMs) +
      " selected_ms=" + std::to_string(p.useLt ? bestMs : baselineMs) +
      " verification=" + verification + " warmup_accuracy=" + warmupAccuracy + verificationDetails +
      (foundCandidate ? algorithmDetails(p.algo) : ""));
  }

#ifdef USE_CUTLASS_FUSED_FFN
  struct FixedProbeGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    FixedProbeGraph() = default;
    FixedProbeGraph(const FixedProbeGraph&) = delete;
    FixedProbeGraph& operator=(const FixedProbeGraph&) = delete;
    ~FixedProbeGraph() {
      if(exec) (void)cudaGraphExecDestroy(exec);
      if(graph) (void)cudaGraphDestroy(graph);
      if(begin) (void)cudaEventDestroy(begin);
      if(end) (void)cudaEventDestroy(end);
    }
    template<typename Launch>
    void capture(cudaStream_t stream, Launch&& launch) {
      CUDA_ERR("B11 graph event",cudaEventCreate(&begin));
      CUDA_ERR("B11 graph event",cudaEventCreate(&end));
      bool capturing = false;
      try {
        CUDA_ERR("B11 graph capture",cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        capturing = true;
        CUDA_ERR("B11 graph event",cudaEventRecordWithFlags(begin,stream,cudaEventRecordExternal));
        launch();
        CUDA_ERR("B11 graph event",cudaEventRecordWithFlags(end,stream,cudaEventRecordExternal));
        cudaError_t status = cudaStreamEndCapture(stream,&graph);
        capturing = false;
        CUDA_ERR("B11 graph capture",status);
        CUDA_ERR("B11 graph instantiate",cudaGraphInstantiate(&exec,graph,0));
      }
      catch(...) {
        // Always leave capture mode before RAII destroys events or scratch buffers.
        // Preserve the original exception; an invalidated capture normally makes
        // EndCapture return an error as well.
        if(capturing) {
          cudaGraph_t abandoned = nullptr;
          (void)cudaStreamEndCapture(stream,&abandoned);
          if(abandoned) (void)cudaGraphDestroy(abandoned);
        }
        throw;
      }
    }
    float run(cudaStream_t stream, int count) {
      CUDA_ERR("B11 graph replay",cudaGraphLaunch(exec,stream));
      CUDA_ERR("B11 graph completion",cudaStreamSynchronize(stream));
      float millis;
      CUDA_ERR("B11 graph timer",cudaEventElapsedTime(&millis,begin,end));
      return millis / count;
    }
  };

  void tuneFixed(Plan& p, cublasHandle_t blas, cudaStream_t stream,
                 int m, int n, int k, const void* a, const void* b, const void* c,
                 const cublas_half_t* alpha, const cublas_half_t* beta, bool accumulate,
                 BaselineKind baselineKind, long long outputStrideElts) {
    if(!fixedReady || !CudaB11Gemm::supportsShape(m,n,k))
      return;
    // The original/Lt decision, including its exactness test, is already final.
    // A fixed kernel must improve that actual winner, not merely the old fallback.
    const std::string previousWinner = p.useLt ? "cuBLASLt" : "cuBLAS";
    const size_t bytes = (size_t)m * n * sizeof(cublas_half_t);
    Temporary t;
    CUDA_ERR("B11 GEMM tune buffer",cudaMalloc(&t.output,bytes));
    enum class Probe { Original, Winner, Fixed };
    auto launch = [&](Probe probe, void* output) {
      if(probe == Probe::Fixed) {
        CUDA_ERR("B11 GEMM probe",CudaB11Gemm::run((const half*)b,(const half*)a,(half*)output,
          m,n,k,accumulate,stream));
      }
      else {
        cublasStatus_t status;
        if(probe == Probe::Winner && p.useLt)
          status = cublasLtMatmul(lt,p.op,alpha,a,p.a,b,p.b,beta,output,p.c,output,p.c,
            &p.algo,workspace,workspaceBytes,stream);
        else if(baselineKind == BaselineKind::StridedBatchedOne)
          status = cublasHgemmStridedBatched(blas,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,alpha,
            (const cublas_half_t*)a,m,(long long)m*k,(const cublas_half_t*)b,k,0LL,
            beta,(cublas_half_t*)output,m,outputStrideElts,1);
        else
          status = cublasHgemm(blas,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,alpha,
            (const cublas_half_t*)a,m,(const cublas_half_t*)b,k,beta,(cublas_half_t*)output,m);
        // These are previously selected or original primitives. Real failures
        // propagate rather than being hidden behind another launch.
        CUBLAS_ERR("B11 GEMM reference probe",status);
      }
    };
    auto evaluateOnce = [&](Probe probe, std::vector<uint16_t>& output) {
      if(accumulate)
        CUDA_ERR("B11 GEMM restore residual",cudaMemcpyAsync(t.output,c,bytes,cudaMemcpyDeviceToDevice,stream));
      launch(probe,t.output);
      CUDA_ERR("B11 GEMM accuracy output",cudaMemcpyAsync(output.data(),t.output,bytes,cudaMemcpyDeviceToHost,stream));
      CUDA_ERR("B11 GEMM accuracy completion",cudaStreamSynchronize(stream));
    };
    // Compare to the original cuBLAS primitive even when Lt won the first stage.
    // Do not accept equal nonfinite outputs; a successful probe must be finite.
    std::vector<uint16_t> expected(bytes/sizeof(uint16_t)), actual(expected.size());
    evaluateOnce(Probe::Original,expected);
    evaluateOnce(Probe::Fixed,actual);
    bool accurate = expected == actual;
    if(accurate) {
      for(uint16_t value : actual) {
        if((value & 0x7c00u) == 0x7c00u) {
          accurate = false;
          break;
        }
      }
    }
    std::string verification = accurate ? "not-timed" : "accuracy-reject";
    std::string details;
    if(accurate) {
      // Prime the actual winner eagerly before capture, including any library
      // lazy initialization, and independently verify its output too.
      evaluateOnce(Probe::Winner,actual);
      accurate = expected == actual;
      if(!accurate) verification = "winner-accuracy-reject";
    }
    if(accurate) {
      constexpr int probeCount = 8;
      Temporary ring;
      CUDA_ERR("B11 graph output ring",cudaMalloc(&ring.output,bytes*probeCount));
      auto ringOutput = [&](int index) { return (char*)ring.output + bytes*index; };
      auto restoreRing = [&]() {
        if(accumulate) {
          for(int index = 0; index < probeCount; index++)
            CUDA_ERR("B11 graph restore residual",cudaMemcpyAsync(ringOutput(index),c,bytes,cudaMemcpyDeviceToDevice,stream));
        }
      };
      // Each graph owns its events and exec, and is destroyed before ring/output.
      // Replays are serialized on this handle's stream. No live activation is modified.
      FixedProbeGraph winnerGraph, fixedGraph;
      winnerGraph.capture(stream,[&]() {
        for(int index = 0; index < probeCount; index++) launch(Probe::Winner,ringOutput(index));
      });
      fixedGraph.capture(stream,[&]() {
        for(int index = 0; index < probeCount; index++) launch(Probe::Fixed,ringOutput(index));
      });
      auto measure = [&](FixedProbeGraph& graph, float& millis) {
        float samples[5];
        for(int repeat = 0; repeat < 6; repeat++) {
          // Restores are outside the graph's embedded timing-event interval.
          // Eight independent destinations prevent beta=1 accumulating across nodes.
          restoreRing();
          float elapsed = graph.run(stream,probeCount);
          if(!std::isfinite(elapsed) || elapsed <= 0.0f) return false;
          if(repeat > 0) samples[repeat-1] = elapsed;
        }
        std::sort(samples,samples+5);
        millis = samples[2];
        return true;
      };
      float winnerChecks[3], fixedChecks[3], ratios[3];
      bool complete = true;
      for(int pair = 0; pair < 3; pair++) {
        bool ok;
        if(pair % 2 == 0)
          ok = measure(winnerGraph,winnerChecks[pair]) && measure(fixedGraph,fixedChecks[pair]);
        else
          ok = measure(fixedGraph,fixedChecks[pair]) && measure(winnerGraph,winnerChecks[pair]);
        if(!ok) { complete = false; break; }
        ratios[pair] = fixedChecks[pair] / winnerChecks[pair];
      }
      if(complete) {
        std::sort(winnerChecks,winnerChecks+3);
        std::sort(fixedChecks,fixedChecks+3);
        std::sort(ratios,ratios+3);
        p.useFixed = ratios[1] < 0.98f;
        verification = p.useFixed ? "pass" : "speed-reject";
        details = " timer=cuda-graph-external-events graph_gemms=8 previous_winner_ms=" + std::to_string(winnerChecks[1]) +
          " fixed_ms=" + std::to_string(fixedChecks[1]) +
          " paired_ratio_min=" + std::to_string(ratios[0]) +
          " paired_ratio_median=" + std::to_string(ratios[1]) +
          " paired_ratio_max=" + std::to_string(ratios[2]);
      }
      else verification = "unavailable";
      // Check every captured output with fresh residuals. This catches aliasing
      // and accidental repeated beta accumulation independently of the timer.
      restoreRing();
      (void)fixedGraph.run(stream,probeCount);
      bool graphAccurate = true;
      for(int index = 0; index < probeCount; index++) {
        CUDA_ERR("B11 graph accuracy output",cudaMemcpyAsync(actual.data(),ringOutput(index),bytes,cudaMemcpyDeviceToHost,stream));
        CUDA_ERR("B11 graph accuracy completion",cudaStreamSynchronize(stream));
        graphAccurate = graphAccurate && expected == actual;
      }
      if(!graphAccurate) {
        p.useFixed = false;
        verification = "graph-accuracy-reject";
      }
      details += std::string(" graph_accuracy=") + (graphAccurate ? "exact" : "reject");
    }
    if(logger) logger->write("Cuda backend: B11 GEMM tuning " + std::to_string(m) + "x" + std::to_string(n) + "x" +
      std::to_string(k) + " beta=" + std::to_string(accumulate ? 1 : 0) +
      " baseline=" + (baselineKind == BaselineKind::Hgemm ? "Hgemm" : "HgemmStridedBatchedOne") +
      " output_stride=" + std::to_string(outputStrideElts) + " previous_winner=" + previousWinner +
      " selected " + (p.useFixed ? "fixed-cutlass-128x128x32-s3" : previousWinner) +
      " verification=" + verification + " warmup_accuracy=" + (accurate ? "exact" : "reject") + details);
  }
#endif

public:
  explicit CudaMatmulTuner(Logger* log, bool enableFixed = false) : logger(log) {
    CUBLAS_ERR("Lt create",cublasLtCreate(&lt));
    cudaError_t status = cudaMalloc(&workspace,workspaceBytes);
    if(status != cudaSuccess) { (void)cublasLtDestroy(lt); lt = nullptr; }
    CUDA_ERR("Lt workspace",status);
    if(enableFixed) {
#ifdef USE_CUTLASS_FUSED_FFN
      status = CudaB11Gemm::prepare();
      fixedReady = status == cudaSuccess;
      if(status == cudaErrorNotSupported) (void)cudaGetLastError();
      else if(status != cudaSuccess) {
        // Constructor exceptions do not run this object's destructor.
        if(workspace) { (void)cudaFree(workspace); workspace = nullptr; }
        if(lt) { (void)cublasLtDestroy(lt); lt = nullptr; }
        CUDA_ERR("B11 GEMM prepare",status);
      }
      if(logger) logger->write(std::string("Cuda backend: B11 fixed GEMM ") +
        (fixedReady ? "prepared (SM120, FP16, CTA128x128x32, stages3)" : "unavailable; retaining cuBLAS/Lt") +
        " status=" + cudaGetErrorString(status));
#else
      if(logger) logger->write("Cuda backend: B11 fixed GEMM unavailable without CUTLASS; retaining cuBLAS/Lt");
#endif
    }
  }
  ~CudaMatmulTuner() {
    plans.clear();
    if(workspace) (void)cudaFree(workspace);
    if(lt) (void)cublasLtDestroy(lt);
  }
  CudaMatmulTuner(const CudaMatmulTuner&) = delete;
  CudaMatmulTuner& operator=(const CudaMatmulTuner&) = delete;

  bool apply(cublasHandle_t blas, cudaStream_t stream, bool warmup, int m, int n, int k,
             const void* a, const void* b, void* c, const cublas_half_t* alpha,
             const cublas_half_t* beta, bool accumulate,
             BaselineKind baselineKind = BaselineKind::Hgemm, long long outputStrideElts = 0) {
    if(m < 128 || n < 128 || k < 128 ||
       ((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b) | reinterpret_cast<uintptr_t>(c)) & 15))
      return false;
    const Key key(m,n,k,accumulate,baselineKind,outputStrideElts);
    auto found = plans.find(key);
    if(found == plans.end()) {
      // No host allocation, descriptor creation or synchronization during capture
      // or normal inference. Unwarmed shapes simply use the existing GEMM.
      if(!warmup) return false;
      auto p = std::make_unique<Plan>();
      tune(*p,blas,stream,m,n,k,a,b,c,alpha,beta,accumulate,baselineKind,outputStrideElts);
#ifdef USE_CUTLASS_FUSED_FFN
      tuneFixed(*p,blas,stream,m,n,k,a,b,c,alpha,beta,accumulate,baselineKind,outputStrideElts);
#endif
      found = plans.emplace(key,std::move(p)).first;
    }
    Plan& p = *found->second;
#ifdef USE_CUTLASS_FUSED_FFN
    if(p.useFixed) {
      // A launch error must propagate. Replaying another GEMM could double-add a
      // residual after a partial write, including during CUDA graph capture.
      CUDA_ERR("B11 GEMM replay",CudaB11Gemm::run((const half*)b,(const half*)a,(half*)c,m,n,k,accumulate,stream));
      return true;
    }
#endif
    if(!p.useLt) return false;
    // Once a validated algorithm is selected, launch failures are real errors.
    // Never rerun after a potentially partial write to a residual activation.
    CUBLAS_ERR("Lt replay",cublasLtMatmul(lt,p.op,alpha,a,p.a,b,p.b,beta,c,p.c,c,p.c,
      &p.algo,workspace,workspaceBytes,stream));
    return true;
  }
};
#endif
