// Fixed b11tf attention for the CUDA SM120 path. See customCudaB11Attention below.
// Eligible only when mask==nullptr, seqLen=361, heads=12, D=32,
// qStride=kvStride=1152. Probability rounding and FP32 MMA sequence are unchanged.
#pragma once
#include "cudaflashmma.cuh"

namespace b11attention {
using namespace flashmma;
#define FMMA_NEG_BIG (-1e30f)
template<bool Swizzle>
__global__ void __launch_bounds__(128)
flashAttention361Kernel(
  const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V,
  half* __restrict__ out, float scale
) {
#if __CUDA_ARCH__ >= 750
  constexpr int D = 32;
  constexpr int BQ = 64;
  constexpr int seqLen = 361;
  constexpr int numHeads = 12;
  constexpr int qStride = 1152;
  constexpr int kvStride = 1152;
  constexpr int BKV = FMMA_BLOCK_KV;
  constexpr int NTHREADS = (BQ / 16) * 32;
  static_assert(BQ == 32 || BQ == 64 || (BQ == 128 && D == 32), "supported query tile");
  // XOR within each 32-half row. For a fixed 8-half column group, the
  // eight consecutive row starts map to eight distinct 16-byte bank groups.
  constexpr int ST = Swizzle ? D : D + 8;
  auto smemOffset = [](int row, int col) {
    return row * ST + (Swizzle ? col ^ (((row >> 1) & 3) * 8) : col);
  };
  constexpr int VEC = 8;         // halfs per vectorized global load (uint4)
  constexpr int KT = D / 16;     // k-tiles along head dim for QK^T
  constexpr int NT_S = BKV / 8;  // n-tiles of the score matrix (8 kv cols each)
  constexpr int KT_PV = BKV / 16;// k-tiles along kv for PV
  constexpr int NT_O = D / 8;    // n-tiles of the output (8 dim cols each)

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int gr = lane >> 2;   // row within 8-row fragment group
  const int q4 = lane & 3;    // quad index -> column pairs

  const int qBlockStart = blockIdx.x * BQ;
  const int h = blockIdx.y;
  const int n = blockIdx.z;
  const int kvh = h;

  // Q/K/V are interior slices of the fixed combined projection output.
  // The attention output remains packed [batch, 361, 384].
  const int oTotalDim = numHeads * D;

  // Double-buffered K/V. Q staging reuses the second K buffer before prefetch.
  // Both layouts preserve cp.async / ldmatrix 16-byte alignment.
  constexpr int KSTAGE_ROWS = BQ > BKV ? BQ : BKV;
  __shared__ __align__(16) half kTiles[2][KSTAGE_ROWS * ST];
  __shared__ __align__(16) half vTiles[2][BKV * ST];
  static_assert(BQ <= KSTAGE_ROWS, "Q staging must fit in buffer 1's K tile");

  const half* Kbase = K + (size_t)n * seqLen * kvStride + kvh * D;
  const half* Vbase = V + (size_t)n * seqLen * kvStride + kvh * D;

  constexpr int KVVECS = BKV * (D / VEC);
  const int numKvTiles = (seqLen + BKV - 1) / BKV;

  // Async-load a K/V tile into the given buffer. Rows past seqLen zero-fill, and their
  // mask bias is -inf so they are excluded from the softmax regardless.
  auto issueTileLoad = [&](int tile, int buf) {
    const int kvStart = tile * BKV;
    #pragma unroll
    for(int t = tid; t < KVVECS; t += NTHREADS) {
      int row = t / (D / VEC);
      int d0 = (t % (D / VEC)) * VEC;
      int g = kvStart + row;
      bool valid = g < seqLen;
      // With valid == false the zero-fill copy should not dereference the source, but keep the
      // address in range anyway rather than pointing up to a tile past the end of K/V.
      size_t off = (size_t)(valid ? g : 0) * kvStride + d0;
      fmmaCpAsync16(&kTiles[buf][smemOffset(row, d0)], Kbase + off, valid);
      fmmaCpAsync16(&vTiles[buf][smemOffset(row, d0)], Vbase + off, valid);
    }
    fmmaCpAsyncCommit();
  };

  // Prologue: start loading kv tile 0, and stage Q into buffer 1's K space (coalesced),
  // zero-filled past seqLen.
  issueTileLoad(0, 0);
  {
    half* qTile = kTiles[1];
    constexpr int QVECS = BQ * (D / VEC);
    #pragma unroll
    for(int t = tid; t < QVECS; t += NTHREADS) {
      int row = t / (D / VEC);
      int d0 = (t % (D / VEC)) * VEC;
      int qPos = qBlockStart + row;
      uint4 val = make_uint4(0, 0, 0, 0);
      if(qPos < seqLen)
        val = *reinterpret_cast<const uint4*>(Q + ((size_t)n * seqLen + qPos) * qStride + h * D + d0);
      *reinterpret_cast<uint4*>(&qTile[smemOffset(row, d0)]) = val;
    }
  }
  __syncthreads();

  // Q fragments in registers (A operand of m16n8k16, rows warp*16 .. warp*16+15).
  uint32_t qA[KT][4];
  {
    const half* qTile = kTiles[1];
    const int r0 = warp * 16 + gr;
    #pragma unroll
    for(int kt = 0; kt < KT; kt++) {
      int d0 = kt * 16 + q4 * 2;
      qA[kt][0] = *reinterpret_cast<const uint32_t*>(&qTile[smemOffset(r0, d0)]);
      qA[kt][1] = *reinterpret_cast<const uint32_t*>(&qTile[smemOffset(r0 + 8, d0)]);
      qA[kt][2] = *reinterpret_cast<const uint32_t*>(&qTile[smemOffset(r0, d0 + 8)]);
      qA[kt][3] = *reinterpret_cast<const uint32_t*>(&qTile[smemOffset(r0 + 8, d0 + 8)]);
    }
  }

  const float scaleLog2 = scale * 1.4426950408889634f;  // fold log2(e) into the scale

  float sFrag[NT_S][4];      // scores, later reused as softmax numerators p
  float oFrag[NT_O][4];      // output accumulator
  float rowMax[2] = {FMMA_NEG_BIG, FMMA_NEG_BIG};  // rows gr and gr+8 of this warp's 16
  float rowSum[2] = {0.0f, 0.0f};
  #pragma unroll
  for(int i = 0; i < NT_O; i++) {
    oFrag[i][0] = 0.0f; oFrag[i][1] = 0.0f; oFrag[i][2] = 0.0f; oFrag[i][3] = 0.0f;
  }

  for(int tile = 0; tile < numKvTiles; tile++) {
    // Wait for this tile's async loads, then start prefetching the next tile. The barrier
    // also guarantees the buffer being written was fully consumed two iterations ago
    // (and, on the first iteration, that all warps have read their Q fragments).
    fmmaCpAsyncWaitAll();
    __syncthreads();
    if(tile + 1 < numKvTiles)
      issueTileLoad(tile + 1, (tile + 1) & 1);

    const half* kTile = kTiles[tile & 1];
    const half* vTile = vTiles[tile & 1];

    // S = Q K^T for this warp's 16 rows x BKV cols. B fragments via ldmatrix: per
    // (kt, ntp) one x4 covers score n-tiles 2*ntp and 2*ntp+1 (kv rows ntp*16..+15,
    // head-dim cols kt*16..+15).
    #pragma unroll
    for(int nt = 0; nt < NT_S; nt++) {
      sFrag[nt][0] = 0.0f; sFrag[nt][1] = 0.0f; sFrag[nt][2] = 0.0f; sFrag[nt][3] = 0.0f;
    }
    #pragma unroll
    for(int kt = 0; kt < KT; kt++) {
      #pragma unroll
      for(int ntp = 0; ntp < NT_S / 2; ntp++) {
        uint32_t b[4];
        const half* rp = &kTile[smemOffset(ntp * 16 + ((lane >> 4) & 1) * 8 + (lane & 7),
                                         kt * 16 + ((lane >> 3) & 1) * 8)];
        fmmaLdmatrixX4(b[0], b[1], b[2], b[3], rp);
        fmmaMma16816(sFrag[2 * ntp], qA[kt], b);
        fmmaMma16816(sFrag[2 * ntp + 1], qA[kt], b + 2);
      }
    }

    // Scale (folded with log2(e) so exp becomes exp2) and add the key mask bias, then
    // find the per-row max of this tile.
    // C fragment layout: c0=(row gr, col 2*q4), c1=(gr, 2*q4+1), c2=(gr+8, 2*q4),
    // c3=(gr+8, 2*q4+1), col offset nt*8 within the tile.
    float pm0[NT_S], pm1[NT_S];
    #pragma unroll
    for(int nt = 0; nt < NT_S; nt++) {
      // Preserve the original score fma and the final tile's -inf padding.
      float b0 = tile * BKV + nt * 8 + q4 * 2 < seqLen ? 0.0f : -CUDART_INF_F;
      float b1 = tile * BKV + nt * 8 + q4 * 2 + 1 < seqLen ? 0.0f : -CUDART_INF_F;
      float s0 = fmaf(sFrag[nt][0], scaleLog2, b0);
      float s1 = fmaf(sFrag[nt][1], scaleLog2, b1);
      float s2 = fmaf(sFrag[nt][2], scaleLog2, b0);
      float s3 = fmaf(sFrag[nt][3], scaleLog2, b1);
      sFrag[nt][0] = s0; sFrag[nt][1] = s1; sFrag[nt][2] = s2; sFrag[nt][3] = s3;
      pm0[nt] = fmaxf(s0, s1);
      pm1[nt] = fmaxf(s2, s3);
    }
    // Tree reduction keeps the dependency chain short.
    #pragma unroll
    for(int w = NT_S / 2; w >= 1; w >>= 1) {
      #pragma unroll
      for(int i = 0; i < w; i++) {
        pm0[i] = fmaxf(pm0[i], pm0[i + w]);
        pm1[i] = fmaxf(pm1[i], pm1[i + w]);
      }
    }
    float tileMax0 = pm0[0];
    float tileMax1 = pm1[0];
    // Row stats live across the 4 lanes of a quad (same gr, different q4).
    tileMax0 = fmaxf(tileMax0, __shfl_xor_sync(0xffffffff, tileMax0, 1));
    tileMax0 = fmaxf(tileMax0, __shfl_xor_sync(0xffffffff, tileMax0, 2));
    tileMax1 = fmaxf(tileMax1, __shfl_xor_sync(0xffffffff, tileMax1, 1));
    tileMax1 = fmaxf(tileMax1, __shfl_xor_sync(0xffffffff, tileMax1, 2));

    // rowMax stays >= -1e30 (its initial value) even when the whole tile is masked
    // (tileMax == -inf), so alpha is well-defined and masked scores give
    // exp2(-inf - rowMax) == 0 exactly.
    const float newMax0 = fmaxf(rowMax[0], tileMax0);
    const float newMax1 = fmaxf(rowMax[1], tileMax1);
    const float alpha0 = fmmaExp2(rowMax[0] - newMax0);
    const float alpha1 = fmmaExp2(rowMax[1] - newMax1);
    rowMax[0] = newMax0;
    rowMax[1] = newMax1;

    // p = exp2(s - rowMax). Masked entries are exactly 0 via s == -inf.
    #pragma unroll
    for(int nt = 0; nt < NT_S; nt++) {
      float p0 = fmmaExp2(sFrag[nt][0] - newMax0);
      float p1 = fmmaExp2(sFrag[nt][1] - newMax0);
      float p2 = fmmaExp2(sFrag[nt][2] - newMax1);
      float p3 = fmmaExp2(sFrag[nt][3] - newMax1);
      sFrag[nt][0] = p0; sFrag[nt][1] = p1; sFrag[nt][2] = p2; sFrag[nt][3] = p3;
      pm0[nt] = p0 + p1;
      pm1[nt] = p2 + p3;
    }
    #pragma unroll
    for(int w = NT_S / 2; w >= 1; w >>= 1) {
      #pragma unroll
      for(int i = 0; i < w; i++) {
        pm0[i] += pm0[i + w];
        pm1[i] += pm1[i + w];
      }
    }
    float tsum0 = pm0[0];
    float tsum1 = pm1[0];
    tsum0 += __shfl_xor_sync(0xffffffff, tsum0, 1);
    tsum0 += __shfl_xor_sync(0xffffffff, tsum0, 2);
    tsum1 += __shfl_xor_sync(0xffffffff, tsum1, 1);
    tsum1 += __shfl_xor_sync(0xffffffff, tsum1, 2);
    rowSum[0] = rowSum[0] * alpha0 + tsum0;
    rowSum[1] = rowSum[1] * alpha1 + tsum1;

    // Rescale the running output by alpha.
    #pragma unroll
    for(int nt = 0; nt < NT_O; nt++) {
      oFrag[nt][0] *= alpha0;
      oFrag[nt][1] *= alpha0;
      oFrag[nt][2] *= alpha1;
      oFrag[nt][3] *= alpha1;
    }

    // Convert P to FP16 A fragments. The QK^T C fragment layout maps directly onto the
    // A fragment layout: score tiles 2*kt2 and 2*kt2+1 form the 16x16 A tile for PV k-tile kt2.
    uint32_t pA[KT_PV][4];
    #pragma unroll
    for(int kt2 = 0; kt2 < KT_PV; kt2++) {
      pA[kt2][0] = fmmaPackHalf2(sFrag[2 * kt2][0], sFrag[2 * kt2][1]);
      pA[kt2][1] = fmmaPackHalf2(sFrag[2 * kt2][2], sFrag[2 * kt2][3]);
      pA[kt2][2] = fmmaPackHalf2(sFrag[2 * kt2 + 1][0], sFrag[2 * kt2 + 1][1]);
      pA[kt2][3] = fmmaPackHalf2(sFrag[2 * kt2 + 1][2], sFrag[2 * kt2 + 1][3]);
    }

    // O += P V. B fragments via transposed ldmatrix: per (c, nt) one x4 covers PV
    // k-tiles 2*c and 2*c+1 (kv rows c*32 + lane, output cols nt*8..+7).
    #pragma unroll
    for(int c = 0; c < KT_PV / 2; c++) {
      #pragma unroll
      for(int nt = 0; nt < NT_O; nt++) {
        uint32_t b[4];
        const half* rp = &vTile[smemOffset(c * 32 + lane, nt * 8)];
        fmmaLdmatrixX4Trans(b[0], b[1], b[2], b[3], rp);
        fmmaMma16816(oFrag[nt], pA[2 * c], b);
        fmmaMma16816(oFrag[nt], pA[2 * c + 1], b + 2);
      }
    }
  }

  // Normalize and write output, with the original row-sum guard and tail check.
  {
    const float inv0 = (rowSum[0] > 0.0f) ? (1.0f / rowSum[0]) : 0.0f;
    const float inv1 = (rowSum[1] > 0.0f) ? (1.0f / rowSum[1]) : 0.0f;
    const int r0 = qBlockStart + warp * 16 + gr;
    const int r1 = r0 + 8;
    if(r0 < seqLen) {
      float f = inv0;
      half* o = out + ((size_t)n * seqLen + r0) * oTotalDim + h * D;
      #pragma unroll
      for(int nt = 0; nt < NT_O; nt++) {
        int d0 = nt * 8 + q4 * 2;
        *reinterpret_cast<uint32_t*>(o + d0) = fmmaPackHalf2(oFrag[nt][0] * f, oFrag[nt][1] * f);
      }
    }
    if(r1 < seqLen) {
      float f = inv1;
      half* o = out + ((size_t)n * seqLen + r1) * oTotalDim + h * D;
      #pragma unroll
      for(int nt = 0; nt < NT_O; nt++) {
        int d0 = nt * 8 + q4 * 2;
        *reinterpret_cast<uint32_t*>(o + d0) = fmmaPackHalf2(oFrag[nt][2] * f, oFrag[nt][3] * f);
      }
    }
  }
#endif  // __CUDA_ARCH__ >= 750
}

#undef FMMA_NEG_BIG
template<bool Swizzle>
inline void launch361(const half* q, const half* k, const half* v, half* out,
                      int batch, cudaStream_t stream) {
  flashAttention361Kernel<Swizzle><<<dim3(6,12,batch),128,0,stream>>>(
    q, k, v, out, 1.0f / sqrtf(32.0f));
}
} // namespace b11attention

// The caller probes the existing MMA kernel once per handle before using this
// helper. Both kernels have the same minimum compiled architecture, so a build
// whose device kernels are empty stubs is rejected before this dispatch.
// There are no runtime device queries, allocations or synchronizations here.
bool customCudaB11Attention(
  const half* q, const half* k, const half* v, const half* mask, half* out,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int vHeadDim,
  int qStride, int kvStride, int majorComputeCapability, int minorComputeCapability,
  cudaStream_t stream
) {
  if(majorComputeCapability != 12 || minorComputeCapability != 0 || mask != nullptr)
    return false;
  if(seqLen != 361 || numHeads != 12 || numKVHeads != 12 || qHeadDim != 32 || vHeadDim != 32)
    return false;
  if(qStride != 1152 || kvStride != 1152 || batchSize <= 0 || batchSize > 65535)
    return false;
  if(q == nullptr || k == nullptr || v == nullptr || out == nullptr)
    return false;
  if((((uintptr_t)q) | ((uintptr_t)k) | ((uintptr_t)v) | ((uintptr_t)out)) & 15)
    return false;
  b11attention::launch361<true>(q, k, v, out, batchSize, stream);
  return true;
}
