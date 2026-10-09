// Opt-in CUDA graph runner for one KataGo compute handle.
// It must be destroyed before the handle's scratch, inputs, outputs, library handles,
// and weights, after synchronizing its stream. Not shared between server threads.
#ifndef NEURALNET_CUDAGRAPH_H_
#define NEURALNET_CUDAGRAPH_H_

#include "../neuralnet/cudaerrorcheck.h"
#include "../core/logger.h"
#include <array>
#include <cstdint>
#include <exception>
#include <memory>
#include <new>

class CudaGraphRunner {
  struct Batch {
    bool warmed = false;
    bool disabled = false;
    cudaGraphExec_t exec = nullptr;
    Batch() = default;
    ~Batch() { if(exec != nullptr) (void)cudaGraphExecDestroy(exec); }
    Batch(const Batch&) = delete;
    Batch& operator=(const Batch&) = delete;
  };
  struct Graph {
    cudaGraph_t graph = nullptr;
    ~Graph() { if(graph != nullptr) (void)cudaGraphDestroy(graph); }
    Graph() = default;
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
  };
  struct AllocationGuard {
    bool& allowed;
    bool saved;
    explicit AllocationGuard(bool& flag) : allowed(flag), saved(flag) { allowed = false; }
    ~AllocationGuard() { allowed = saved; }
  };

  const bool enabled;
  const int maxBatchSize;
  const int maxCachedBatches;
  const int numVariants;
  Logger* logger;
  // The maximum batch is fixed at handle construction. Indexing directly keeps
  // replay free of map lookups and host allocations. Device graph storage is
  // bounded separately, since a long search can encounter many rare shapes.
  std::unique_ptr<Batch[]> batches;
  int numCachedBatches;
  bool loggedCapacityLimit;
  int executionPlanKey;
  uint64_t hostAllocationIdentity;
  std::array<const void*,8> hostBindings;

  static bool isCaptureRejection(cudaError_t status) {
    return status == cudaErrorStreamCaptureUnsupported ||
      status == cudaErrorStreamCaptureInvalidated || status == cudaErrorNotSupported;
  }

  void disable(Batch& state, int batchSize, int variant, const std::string& reason) {
    state.disabled = true;
    if(logger != nullptr)
      logger->write("Cuda backend: CUDA Graph disabled for batch " +
        Global::intToString(batchSize) + ", variant " + Global::intToString(variant) +
        ", using eager inference: " + reason);
  }

public:
  CudaGraphRunner(bool enable, int maxBatch, int maxCached, Logger* log, int variants = 1)
    : enabled(enable), maxBatchSize(maxBatch), maxCachedBatches(maxCached), numVariants(variants), logger(log),
      batches(enable ? std::make_unique<Batch[]>(((size_t)maxBatch + 1) * variants) : nullptr),
      numCachedBatches(0), loggedCapacityLimit(false), executionPlanKey(-1),
      hostAllocationIdentity(0), hostBindings{}
  {
    testAssert(variants > 0);
  }
  CudaGraphRunner(const CudaGraphRunner&) = delete;
  CudaGraphRunner& operator=(const CudaGraphRunner&) = delete;

  // Called only after getOutput's final stream synchronize succeeded. A complete,
  // successful eager execution is required before capturing a batch shape.
  void markCompleted(int batchSize, int variant = 0) {
    if(enabled && batchSize > 0 && batchSize <= maxBatchSize) {
      testAssert(variant >= 0 && variant < numVariants);
      batches[(size_t)batchSize * numVariants + variant].warmed = true;
    }
  }

  // InputBuffers owns the pinned host allocations. Its identity is monotonically
  // assigned, so replacing buffers is detected even if malloc reuses every address.
  // All previous getOutput calls have synchronized before this method is reached.
  // Captured memcpy nodes read the current host contents on every graph launch.
  void bindTransferBuffers(uint64_t allocationIdentity, const std::array<const void*,8>& bindings) {
    if(!enabled)
      return;
    if(hostAllocationIdentity != allocationIdentity || hostBindings != bindings) {
      clear();
      hostAllocationIdentity = allocationIdentity;
      hostBindings = bindings;
    }
  }

  // The owning ComputeHandle synchronizes before this explicit teardown.
  void clear() {
    if(!enabled)
      return;
    for(size_t i = (size_t)numVariants; i < ((size_t)maxBatchSize + 1) * numVariants; i++) {
      if(batches[i].exec != nullptr) {
        (void)cudaGraphExecDestroy(batches[i].exec);
        batches[i].exec = nullptr;
      }
      batches[i].warmed = false;
      batches[i].disabled = false;
    }
    numCachedBatches = 0;
    loggedCapacityLimit = false;
  }

  template<class Enqueue>
  void run(int batchSize, bool isWarmup, int currentExecutionPlanKey, cudaStream_t stream,
           bool& allowNewScratchAllocations, const Enqueue& enqueue, int variant = 0) {
    // Warmup must finish for ALL batches first: cuDNN SDPA can disable itself
    // handle-wide after a rejected shape. Do not capture a soon-to-change path.
    if(!enabled) {
      enqueue();
      return;
    }
    if(batchSize <= 0 || batchSize > maxBatchSize)
      throw StringError("CUDA Graph batch size exceeds this compute handle's capacity");
    if(variant < 0 || variant >= numVariants)
      throw StringError("CUDA Graph variant exceeds this compute handle's capacity");
    // A new SDPA shape can turn off that backend handle-wide. Graphs captured
    // earlier must not keep replaying the old path after such a transition.
    // getOutput synchronizes every prior call before reaching this point.
    if(executionPlanKey != currentExecutionPlanKey) {
      clear();
      executionPlanKey = currentExecutionPlanKey;
    }
    if(isWarmup) {
      enqueue();
      return;
    }
    Batch& state = batches[(size_t)batchSize * numVariants + variant];
    if(state.exec != nullptr) {
      // A launch/runtime failure is a real inference error. Never rerun a graph
      // that might have partly executed and hide the failure behind fallback.
      CUDA_ERR("CUDA Graph launch", cudaGraphLaunch(state.exec, stream));
      return;
    }
    if(state.disabled || !state.warmed) {
      enqueue();
      return;
    }
    if(numCachedBatches >= maxCachedBatches) {
      if(!loggedCapacityLimit && logger != nullptr)
        logger->write("Cuda backend: CUDA Graph cache limit reached; additional batch sizes use eager inference");
      loggedCapacityLimit = true;
      enqueue();
      return;
    }

    // Cold capture only. Drain preceding copies and surface existing async
    // failures, keeping them separate from recoverable capture rejections.
    CUDA_ERR("CUDA Graph prepare", cudaStreamSynchronize(stream));
    CUDA_ERR("CUDA Graph prepare", cudaPeekAtLastError());
    cudaError_t beginStatus = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if(beginStatus != cudaSuccess) {
      if(!isCaptureRejection(beginStatus))
        CUDA_ERR("CUDA Graph begin", beginStatus);
      cudaError_t pending = cudaGetLastError();
      if(pending != cudaSuccess && !isCaptureRejection(pending))
        CUDA_ERR("CUDA Graph begin", pending);
      disable(state, batchSize, variant, cudaGetErrorString(beginStatus));
      enqueue();
      return;
    }

    Graph captured;
    std::exception_ptr captureFailure;
    {
      AllocationGuard guard(allowNewScratchAllocations);
      try {
        enqueue();
      }
      catch(...) {
        captureFailure = std::current_exception();
      }
    }
    // This MUST run after every successful BeginCapture, including exceptions.
    // EndCapture clears the stream's invalidated capture state as well.
    cudaError_t endStatus = cudaStreamEndCapture(stream, &captured.graph);
    if(endStatus != cudaSuccess && !isCaptureRejection(endStatus))
      CUDA_ERR("CUDA Graph end", endStatus);
    cudaError_t pending = cudaGetLastError();
    if(pending != cudaSuccess && !isCaptureRejection(pending))
      CUDA_ERR("CUDA Graph capture", pending);
    CUDA_ERR("CUDA Graph capture cleanup", cudaStreamSynchronize(stream));

    std::string failureReason;
    if(captureFailure) {
      try {
        std::rethrow_exception(captureFailure);
      }
      catch(const std::bad_alloc&) {
        throw; // Host out-of-memory is not a graph capability failure.
      }
      catch(const std::exception& e) {
        failureReason = e.what();
      }
      // Non-standard exceptions propagate after EndCapture and cleanup above.
    }
    if(endStatus != cudaSuccess)
      failureReason += std::string("; EndCapture: ") + cudaGetErrorString(endStatus);
    if(captured.graph == nullptr && failureReason.empty())
      failureReason = "capture returned no graph";
    if(!failureReason.empty()) {
      disable(state, batchSize, variant, failureReason);
      // Capture executes no model kernels. Retry once outside capture; a real
      // model/library error still propagates through this ordinary eager call.
      enqueue();
      return;
    }

#if CUDART_VERSION >= 11040
    cudaError_t instantiateStatus = cudaGraphInstantiateWithFlags(&state.exec, captured.graph, 0);
#else
    cudaError_t instantiateStatus = cudaGraphInstantiate(&state.exec, captured.graph, nullptr, nullptr, 0);
#endif
    if(instantiateStatus != cudaSuccess) {
      // Graph capacity/unsupported graph structures can fail even when capture
      // succeeded. No graph has run; keeping eager inference is safe.
      if(instantiateStatus != cudaErrorMemoryAllocation &&
         instantiateStatus != cudaErrorInvalidValue && !isCaptureRejection(instantiateStatus))
        CUDA_ERR("CUDA Graph instantiate", instantiateStatus);
      if(state.exec != nullptr) {
        (void)cudaGraphExecDestroy(state.exec);
        state.exec = nullptr;
      }
      pending = cudaGetLastError();
      if(pending != cudaSuccess && pending != instantiateStatus && !isCaptureRejection(pending))
        CUDA_ERR("CUDA Graph instantiate", pending);
      CUDA_ERR("CUDA Graph instantiate cleanup", cudaStreamSynchronize(stream));
      disable(state, batchSize, variant, cudaGetErrorString(instantiateStatus));
      enqueue();
      return;
    }

    numCachedBatches++;
    if(logger != nullptr)
      logger->write("Cuda backend: captured CUDA Graph for batch " + Global::intToString(batchSize) +
        ", variant " + Global::intToString(variant));
    CUDA_ERR("CUDA Graph first launch", cudaGraphLaunch(state.exec, stream));
  }
};

#endif
