/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
// Adapted from CUTLASS examples/45_dual_gemm/kernel/dual_gemm.h.
// KataGo specialization: no split-K or intermediate stores; SwiGLU is
// applied to accumulator fragments before a single output epilogue.
#pragma once

// KataGo-only no-source/no-split-K FFN specialization. Gemm is the original
// FP16 DualGemm (CTA128x64x32, warp64x32x32, three pipeline stages).
// The two accumulator fragments have identical element layouts. Applying the
// original eight-half SwiGLU before rearranging them preserves each operation
// and its rounding, while only one fragment is staged through shared memory.
namespace CudaFFNRegisterEpilogue {
template<class Gemm>
struct Kernel {
  using Original = typename Gemm::DualGemmKernel;
  using Params = typename Original::Params;
  using Mma = typename Gemm::DualMma;
  using Epilogue = typename Gemm::Epilogue0;
  using Identity = typename Gemm::EpilogueOutputOp0;
  using SwiGLU = typename Gemm::EpilogueOutputOp2;
  using Swizzle = typename Gemm::ThreadblockSwizzle;
  static constexpr int kThreadCount = Original::kThreadCount;
  static_assert(!Gemm::kSplitKSerial && !Gemm::kStoreD0 && !Gemm::kStoreD1,
                "register FFN requires no split K or intermediate output");
  static_assert(Epilogue::kPartitionsK == 1,"activation follows the full K reduction");
  static_assert(Mma::FragmentC::kElements % 8 == 0,"eight-half SwiGLU");
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  CUTLASS_DEVICE void operator()(Params const& p,SharedStorage& shared) {
    auto tile=Swizzle().get_tile_offset(p.swizzle_log_tile);
    if(tile.m()>=p.grid_tiled_shape.m() || tile.n()>=p.grid_tiled_shape.n()) return;
    const int thread=threadIdx.x;
    const int warp=__shfl_sync(0xffffffff,thread/32,0);
    const int lane=thread%32;
    typename Mma::IteratorA a(p.params_A0,p.ref_A0.data(),
      {p.problem_size.m(),p.problem_size.k()},thread,{tile.m()*128,0});
    typename Mma::IteratorB0 b0(p.params_B0,p.ref_B0.data(),
      {p.problem_size.k(),p.problem_size.n()},thread,{0,tile.n()*64});
    typename Mma::IteratorB1 b1(p.params_B1,p.ref_B1.data(),
      {p.problem_size.k(),p.problem_size.n()},thread,{0,tile.n()*64});
    typename Mma::FragmentC left,gate;
    left.clear(); gate.clear();
    Mma mma(shared.main_loop,thread,warp,lane);
    mma((p.problem_size.k()+31)/32,left,gate,a,b0,b1,left,gate);

    SwiGLU activation{typename SwiGLU::Params{}};
    CUTLASS_PRAGMA_UNROLL
    for(int i=0;i<Mma::FragmentC::kElements;i+=8) {
      typename SwiGLU::FragmentAccumulator l,g;
      CUTLASS_PRAGMA_UNROLL
      for(int j=0;j<8;j++) { l[j]=left[i+j]; g[j]=gate[i+j]; }
      auto result=activation(l,g);
      CUTLASS_PRAGMA_UNROLL
      for(int j=0;j<8;j++) left[i+j]=result[j];
    }
    typename Epilogue::OutputTileIterator destination(
      p.params_D2,p.ref_D2.data(),p.problem_size.mn(),thread,{tile.m()*128,tile.n()*64});
    Epilogue epilogue(shared.epilogue,thread,warp,lane);
    epilogue(Identity{typename Identity::Params{}},destination,left);
  }
};
}
