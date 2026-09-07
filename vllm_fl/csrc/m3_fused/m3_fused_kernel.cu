// Copyright (c) 2026 BAAI. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Native build of upstream vLLM's MiniMax-M3 fused attention-preprocessing
// kernel (csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu)
// for the FL stack, adapted to plain torch APIs:
//   - PDL (cudaLaunchKernelEx / programmatic launch) dropped: plain <<<>>>
//   - ROCm branches dropped
//   - libtorch_stable types -> at::Tensor / TORCH_CHECK
//   - unquantized ("auto") KV-cache path only (M3's config); fp8 e4m3 index
//     outputs kept
// Registers ``_C::fused_minimax_m3_qknorm_rope_kv_insert`` — the same op the
// pure-torch fallback registers, so the python side steps aside when this
// module is importable.

#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define FINAL_MASK 0xffffffffu

namespace vllm {
namespace m3_fused {

constexpr int kHeadDim = 128;
constexpr int kNumLanes = 32;
constexpr int kElemsPerLane = kHeadDim / kNumLanes;  // 4

// ── minimal type shims (from csrc/type_convert.cuh) ─────────────────────────
template <typename T>
struct _tc;  // convert scalar_t <-> float / packed2 <-> float2

template <>
struct _tc<__half> {
  using packed = __half2;
  static __device__ __forceinline__ float2 convert2(__half2 x) {
    return __half22float2(x);
  }
  static __device__ __forceinline__ __half2 convert2(float2 x) {
    return __float22half2_rn(x);
  }
};

template <>
struct _tc<__nv_bfloat16> {
  using packed = __nv_bfloat162;
  static __device__ __forceinline__ float2 convert2(__nv_bfloat162 x) {
    return __bfloat1622float2(x);
  }
  static __device__ __forceinline__ __nv_bfloat162 convert2(float2 x) {
    return __float22bfloat162_rn(x);
  }
};

// ── helpers (verbatim from the upstream kernel) ─────────────────────────────
__device__ __forceinline__ float warpReduceSum(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  }
  return val;
}

// Gemma RMSNorm over the full head, rounded back to scalar_t like the
// materialized unfused norm output, followed by partial NeoX RoPE on the
// leading ``rotary_dim`` dims. Each lane owns ``kElemsPerLane`` contiguous
// dims [laneId*4, laneId*4+4).
template <typename scalar_t>
__device__ __forceinline__ void normAndRope(
    float (&elems)[kElemsPerLane], int const laneId, float const eps,
    scalar_t const* __restrict__ weight, bool const do_rope,
    int const rotary_dim, scalar_t const* __restrict__ cos_ptr,
    bool const apply_norm) {
  if (apply_norm) {
    float sumsq = 0.0f;
#pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) sumsq += elems[i] * elems[i];
    sumsq = warpReduceSum(sumsq);
    float const rms_rcp = rsqrtf(sumsq / static_cast<float>(kHeadDim) + eps);
#pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
      int const dim = laneId * kElemsPerLane + i;
      float const w = 1.0f + static_cast<float>(weight[dim]);
      elems[i] = elems[i] * rms_rcp * w;
    }
  }

  if (do_rope) {
    int const half = rotary_dim / 2;
    int const dim0 = laneId * kElemsPerLane;
    bool const in_rope = dim0 < rotary_dim;
    int const lane_xor = half / kElemsPerLane;  // partner-lane distance

    float partner[kElemsPerLane];
#pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
      partner[i] = __shfl_xor_sync(FINAL_MASK, elems[i], lane_xor, 32);
    }
    if (in_rope) {
      bool const first_half = dim0 < half;
      int const i_base = first_half ? dim0 : (dim0 - half);
      scalar_t const* sin_ptr = cos_ptr + half;
#pragma unroll
      for (int i = 0; i < kElemsPerLane; i++) {
        float const c = static_cast<float>(cos_ptr[i_base + i]);
        float const s = static_cast<float>(sin_ptr[i_base + i]);
        if (first_half) {
          elems[i] = elems[i] * c - partner[i] * s;
        } else {
          elems[i] = elems[i] * c + partner[i] * s;
        }
      }
    }
  }
}

// Vectorized 4-elem (uint2) load/store, verbatim from the upstream kernel —
// the PPU toolchain compiles this exact form for the vendor's native build.
template <typename scalar_t>
__device__ __forceinline__ void loadElems(
    scalar_t const* __restrict__ src, float (&elems)[kElemsPerLane]) {
  using packed_t = typename _tc<scalar_t>::packed;
  uint2 v = *reinterpret_cast<uint2 const*>(src);
  auto const* p = reinterpret_cast<packed_t const*>(&v);
#pragma unroll
  for (int i = 0; i < kElemsPerLane / 2; i++) {
    float2 f2 = _tc<scalar_t>::convert2(p[i]);
    elems[2 * i] = f2.x;
    elems[2 * i + 1] = f2.y;
  }
}

template <typename scalar_t>
__device__ __forceinline__ void storeElems(
    scalar_t* __restrict__ dst, float const (&elems)[kElemsPerLane]) {
  using packed_t = typename _tc<scalar_t>::packed;
  uint2 v;
  auto* p = reinterpret_cast<packed_t*>(&v);
#pragma unroll
  for (int i = 0; i < kElemsPerLane / 2; i++) {
    p[i] = _tc<scalar_t>::convert2(make_float2(elems[2 * i], elems[2 * i + 1]));
  }
  *reinterpret_cast<uint2*>(dst) = v;
}

// Software fp32 -> e4m3 (satfinite, RNE) encoder: the PPU toolchain
// miscompiles the __nv_cvt intrinsic, so emit the byte via integer ops
// (bit-exact vs torch's .to(float8_e4m3fn) across a 4.2M-value sweep).
__device__ __forceinline__ uint8_t f32ToE4M3(float xf) {
  xf = fminf(fmaxf(xf, -448.0f), 448.0f);
  uint32_t u_all = __float_as_uint(xf);
  uint8_t sign = static_cast<uint8_t>((u_all >> 31) << 7);
  uint32_t u = u_all & 0x7FFFFFFFu;
  int e_f = static_cast<int>((u >> 23) & 255u) - 127;
  uint32_t m_full = (u & 0x7FFFFFu) | 0x800000u;  // implicit leading 1

  // normal: 3-bit mantissa, RNE on the dropped 20 bits
  uint32_t mant = (m_full >> 20) & 7u;
  uint32_t rem = m_full & 0xFFFFFu;
  uint32_t half = 1u << 19;
  bool round_up = (rem > half) || ((rem == half) && ((mant & 1u) == 1u));
  mant += round_up ? 1u : 0u;  // may carry into the exponent
  int e4n = e_f + 7 + static_cast<int>(mant >> 3);
  mant &= 7u;
  uint8_t normal_bits = static_cast<uint8_t>((e4n << 3) | mant);
  bool is_normal = (e_f >= -6) && (e4n <= 15);

  // subnormal / zero: round(|x| * 2^9); 8 carries into the smallest normal.
  float ax = fabsf(xf);
  int sub = static_cast<int>(ax * 512.0f + 0.5f);
  uint8_t sub_bits = static_cast<uint8_t>(sub > 8 ? 8 : sub);

  return sign | (is_normal ? normal_bits : sub_bits);
}

// Store 4 fp32 -> 4 E4M3 bytes (saturating to ±448), packed into one uint32
// store: scalar per-byte stores are miscompiled by the PPU toolchain (same
// as scalar bf16 stores). No scale: RMSNorm outputs are O(1) and the score
// path only needs relative block ordering.
__device__ __forceinline__ void storeElemsFp8(
    uint8_t* __restrict__ dst, float const (&elems)[kElemsPerLane]) {
  uint32_t packed = 0u;
#pragma unroll
  for (int i = 0; i < kElemsPerLane; i++) {
    packed |= static_cast<uint32_t>(f32ToE4M3(elems[i])) << (8 * i);
  }
  *reinterpret_cast<uint32_t*>(dst) = packed;
}

// ── kernel (verbatim modulo PDL) ────────────────────────────────────────────
template <typename scalar_t, typename out_idx_t, bool kIsSparse,
          bool kInsertKV, bool kFp8Idx>
__global__ void fusedMiniMaxM3QNormRopeKVInsertKernel(
    scalar_t* __restrict__ qkv,
    scalar_t* __restrict__ q_out,
    out_idx_t* __restrict__ index_q_out,
    scalar_t const* __restrict__ q_norm_w,
    scalar_t const* __restrict__ k_norm_w,
    scalar_t const* __restrict__ iq_norm_w,
    scalar_t const* __restrict__ ik_norm_w,
    scalar_t const* __restrict__ cos_sin_cache,
    int64_t const* __restrict__ positions,
    int64_t const* __restrict__ slot_mapping,
    int64_t const* __restrict__ index_slot_mapping,
    scalar_t* __restrict__ kv_cache,
    out_idx_t* __restrict__ index_cache,
    float const eps, int const rotary_dim, int const num_tokens,
    int const nq, int const nkv, int const niq, int const block_size,
    int64_t const kv_s_block, int64_t const kv_s_kv, int64_t const kv_s_token,
    int64_t const kv_s_head) {
  int const warpsPerBlock = blockDim.x / 32;
  int const laneId = threadIdx.x % 32;
  int const globalWarpIdx = blockIdx.x * warpsPerBlock + (threadIdx.x / 32);

  int const v_slots = kInsertKV ? nkv : 0;
  int const idx_slots = kIsSparse ? niq + 1 : 0;
  int const slots_per_token = nq + nkv + v_slots + idx_slots;

  int const tokenIdx = globalWarpIdx / slots_per_token;
  int const slot = globalWarpIdx % slots_per_token;
  if (tokenIdx >= num_tokens) return;

  int const k_begin = nq;
  int const v_begin = nq + nkv;
  int const iq_begin = nq + nkv + v_slots;
  int const ik_slot = iq_begin + niq;

  bool const isQ = slot < k_begin;
  bool const isK = slot >= k_begin && slot < v_begin;
  bool isV = false;
  if constexpr (kInsertKV) isV = slot >= v_begin && slot < v_begin + nkv;
  bool isIQ = false, isIK = false;
  if constexpr (kIsSparse) {
    isIQ = slot >= iq_begin && slot < ik_slot;
    isIK = slot == ik_slot;
  }

  int const dim_base = laneId * kElemsPerLane;
  int const qkv_row = (nq + 2 * nkv + (kIsSparse ? (niq + 1) : 0)) * kHeadDim;

  scalar_t* row_ptr = nullptr;
  scalar_t const* norm_w = nullptr;
  bool do_rope = true;
  int head = 0;

  if (isQ) {
    row_ptr =
        qkv + static_cast<int64_t>(tokenIdx) * qkv_row + slot * kHeadDim;
    norm_w = q_norm_w;
  } else if (isK) {
    head = slot - k_begin;
    row_ptr =
        qkv + static_cast<int64_t>(tokenIdx) * qkv_row + slot * kHeadDim;
    norm_w = k_norm_w;
  } else if (isV) {
    head = slot - v_begin;
    row_ptr =
        qkv + static_cast<int64_t>(tokenIdx) * qkv_row + slot * kHeadDim;
    norm_w = nullptr;
    do_rope = false;
  } else if (isIQ) {
    int const ih = slot - iq_begin;
    row_ptr = qkv + static_cast<int64_t>(tokenIdx) * qkv_row +
              (nq + 2 * nkv + ih) * kHeadDim;
    norm_w = iq_norm_w;
  } else {  // isIK
    row_ptr = qkv + static_cast<int64_t>(tokenIdx) * qkv_row +
              (nq + 2 * nkv + niq) * kHeadDim;
    norm_w = ik_norm_w;
  }

  scalar_t* store_ptr = row_ptr;
  if (isQ && q_out != nullptr) {
    store_ptr =
        q_out + static_cast<int64_t>(tokenIdx) * nq * kHeadDim + slot * kHeadDim;
  } else if (isIQ && index_q_out != nullptr) {
    if constexpr (!kFp8Idx) {
      store_ptr = index_q_out +
                  static_cast<int64_t>(tokenIdx) * niq * kHeadDim +
                  (slot - iq_begin) * kHeadDim;
    }
  }

  float elems[kElemsPerLane];
  loadElems<scalar_t>(row_ptr + dim_base, elems);

  if (!isV) {
    int64_t const pos = positions[tokenIdx];
    scalar_t const* cos_ptr = cos_sin_cache + pos * rotary_dim;
    normAndRope<scalar_t>(elems, laneId, eps, norm_w, do_rope, rotary_dim,
                          cos_ptr, /*apply_norm=*/norm_w != nullptr);
    if constexpr (kFp8Idx) {
      if (isIQ && index_q_out != nullptr) {
        storeElemsFp8(index_q_out +
                          static_cast<int64_t>(tokenIdx) * niq * kHeadDim +
                          (slot - iq_begin) * kHeadDim + dim_base,
                      elems);
      } else {
        storeElems<scalar_t>(store_ptr + dim_base, elems);
      }
    } else {
      storeElems<scalar_t>(store_ptr + dim_base, elems);
    }
  }

  if constexpr (kInsertKV) {
    int64_t const sm = (isK || isV)
                           ? slot_mapping[tokenIdx]
                           : (isIK ? index_slot_mapping[tokenIdx] : -1);
    if (sm >= 0) {
      if (isIK) {
        if constexpr (kFp8Idx) {
          storeElemsFp8(index_cache + sm * kHeadDim + dim_base, elems);
        } else {
          storeElems<scalar_t>(index_cache + sm * kHeadDim + dim_base, elems);
        }
      } else if (isK || isV) {
        int64_t const b = sm / block_size;
        int64_t const t = sm % block_size;
        int const kv = isK ? 0 : 1;
        int64_t const off =
            b * kv_s_block + kv * kv_s_kv + t * kv_s_token + head * kv_s_head;
        storeElems<scalar_t>(kv_cache + off + dim_base, elems);
      }
    }
  }
}

// ── launcher ────────────────────────────────────────────────────────────────
template <typename scalar_t>
void launchFusedMiniMaxM3(
    scalar_t* qkv, scalar_t* q_out, void* index_q_out, scalar_t const* q_norm_w,
    scalar_t const* k_norm_w, scalar_t const* iq_norm_w,
    scalar_t const* ik_norm_w, scalar_t const* cos_sin_cache,
    int64_t const* positions, int64_t const* slot_mapping,
    int64_t const* index_slot_mapping, scalar_t* kv_cache, void* index_cache,
    float const eps, int const rotary_dim, int const num_tokens, int const nq,
    int const nkv, int const niq, int const block_size,
    int64_t const kv_s_block, int64_t const kv_s_kv, int64_t const kv_s_token,
    int64_t const kv_s_head, bool const has_index, bool const insert_kv,
    bool const fp8_idx, cudaStream_t stream) {
  int const v_slots = insert_kv ? nkv : 0;
  int const idx_slots = has_index ? niq + 1 : 0;
  int const slots_per_token = nq + nkv + v_slots + idx_slots;

  constexpr int kBlockSize = 256;
  constexpr int kWarpsPerBlock = kBlockSize / 32;
  int64_t const total_warps = static_cast<int64_t>(num_tokens) * slots_per_token;
  int const grid =
      static_cast<int>((total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
  if (grid == 0) return;

#define LAUNCH(IS_SPARSE, INSERT, FP8, OUT_T)                                 \
  fusedMiniMaxM3QNormRopeKVInsertKernel<scalar_t, OUT_T, IS_SPARSE, INSERT,   \
                                        FP8><<<grid, kBlockSize, 0, stream>>>( \
      qkv, q_out, reinterpret_cast<OUT_T*>(index_q_out), q_norm_w, k_norm_w,  \
      iq_norm_w, ik_norm_w, cos_sin_cache, positions, slot_mapping,           \
      index_slot_mapping, kv_cache, reinterpret_cast<OUT_T*>(index_cache),    \
      eps, rotary_dim, num_tokens, nq, nkv, niq, block_size, kv_s_block,      \
      kv_s_kv, kv_s_token, kv_s_head)

  if (has_index) {
    if (insert_kv) {
      if (fp8_idx) {
        LAUNCH(true, true, true, uint8_t);
      } else {
        LAUNCH(true, true, false, scalar_t);
      }
    } else {
      if (fp8_idx) {
        LAUNCH(true, false, true, uint8_t);
      } else {
        LAUNCH(true, false, false, scalar_t);
      }
    }
  } else {
    LAUNCH(false, false, false, scalar_t);
  }
#undef LAUNCH
}

// ── torch op host wrapper (auto/unquantized KV cache only) ─────────────────
void fused_minimax_m3_qknorm_rope_kv_insert(
    at::Tensor& qkv, at::Tensor const& q_norm_weight,
    at::Tensor const& k_norm_weight, at::Tensor const& cos_sin_cache,
    at::Tensor const& positions, int64_t num_heads, int64_t num_kv_heads,
    int64_t rotary_dim, double eps,
    std::optional<at::Tensor> index_q_norm_weight,
    std::optional<at::Tensor> index_k_norm_weight, int64_t num_index_heads,
    std::optional<at::Tensor> slot_mapping,
    std::optional<at::Tensor> index_slot_mapping,
    std::optional<at::Tensor> kv_cache, std::optional<at::Tensor> index_cache,
    int64_t block_size, std::optional<at::Tensor> q_out,
    std::optional<at::Tensor> index_q_out, const std::string& kv_cache_dtype) {
  TORCH_CHECK(qkv.is_cuda() && qkv.is_contiguous(),
              "qkv must be contiguous CUDA");
  TORCH_CHECK(qkv.scalar_type() == at::ScalarType::Half ||
                  qkv.scalar_type() == at::ScalarType::BFloat16,
              "qkv must be float16 or bfloat16");
  TORCH_CHECK(positions.is_cuda() &&
                  positions.scalar_type() == at::ScalarType::Long,
              "positions must be int64 CUDA");
  TORCH_CHECK(cos_sin_cache.is_cuda() && cos_sin_cache.is_contiguous(),
              "cos_sin_cache must be contiguous CUDA");
  TORCH_CHECK(cos_sin_cache.scalar_type() == qkv.scalar_type(),
              "cos_sin_cache dtype must match qkv");
  TORCH_CHECK(cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == rotary_dim,
              "cos_sin_cache shape [max_pos, rotary_dim]");
  TORCH_CHECK(q_norm_weight.scalar_type() == qkv.scalar_type() &&
                  k_norm_weight.scalar_type() == qkv.scalar_type(),
              "q/k norm weight dtype must match qkv");
  TORCH_CHECK(q_norm_weight.numel() == kHeadDim &&
                  k_norm_weight.numel() == kHeadDim,
              "q/k norm weight must have 128 elements");
  TORCH_CHECK(rotary_dim > 0 && rotary_dim % 8 == 0 && rotary_dim <= kHeadDim,
              "rotary_dim must be a positive multiple of 8 and <= 128");

  int const num_tokens = static_cast<int>(qkv.size(0));
  int const nq = static_cast<int>(num_heads);
  int const nkv = static_cast<int>(num_kv_heads);
  int const niq = static_cast<int>(num_index_heads);

  bool const has_index = niq > 0;
  bool const insert_kv = kv_cache.has_value();
  TORCH_CHECK(!insert_kv || kv_cache_dtype == "auto",
              "native M3 fused op supports the unquantized (auto) KV cache");
  int const expected_row =
      (nq + 2 * nkv + (has_index ? niq + 1 : 0)) * kHeadDim;
  TORCH_CHECK(qkv.size(1) == expected_row,
              "qkv last dim must be (num_heads + 2*num_kv_heads"
              " + num_index_heads + 1) * 128 for sparse, "
              "(num_heads + 2*num_kv_heads) * 128 for dense");
  TORCH_CHECK(!insert_kv || has_index,
              "insert mode (kv_cache) requires the index branch (sparse)");
  if (has_index) {
    TORCH_CHECK(index_q_norm_weight.has_value() &&
                    index_k_norm_weight.has_value(),
                "index branch requires both index norm weights");
    TORCH_CHECK(index_q_norm_weight->scalar_type() == qkv.scalar_type() &&
                    index_k_norm_weight->scalar_type() == qkv.scalar_type(),
                "index norm weights dtype must match qkv");
    TORCH_CHECK(index_q_norm_weight->numel() == kHeadDim &&
                    index_k_norm_weight->numel() == kHeadDim,
                "index norm weights must have 128 elements");
  }

  int64_t kv_s_block = 0, kv_s_kv = 0, kv_s_token = 0, kv_s_head = 0;
  at::Tensor const* effective_index_slot_mapping = nullptr;
  if (insert_kv) {
    TORCH_CHECK(slot_mapping.has_value() && slot_mapping->is_cuda() &&
                    slot_mapping->scalar_type() == at::ScalarType::Long,
                "insert mode requires int64 CUDA slot_mapping");
    TORCH_CHECK(!index_slot_mapping.has_value() ||
                    (index_slot_mapping->is_cuda() &&
                     index_slot_mapping->scalar_type() ==
                         at::ScalarType::Long &&
                     index_slot_mapping->numel() == slot_mapping->numel()),
                "index_slot_mapping must be int64 CUDA with slot_mapping "
                "length");
    TORCH_CHECK(kv_cache->scalar_type() == qkv.scalar_type(),
                "auto kv_cache dtype must match qkv");
    TORCH_CHECK(index_cache.has_value() &&
                    (index_cache->scalar_type() == qkv.scalar_type() ||
                     index_cache->scalar_type() ==
                         at::ScalarType::Float8_e4m3fn),
                "insert mode requires index_cache matching qkv dtype or fp8 "
                "e4m3");
    TORCH_CHECK(kv_cache->dim() == 5 && kv_cache->stride(4) == 1,
                "kv_cache must be [nb,2,bs,nkv,head_dim] with contiguous "
                "head_dim (stride(4)==1)");
    kv_s_block = kv_cache->stride(0);
    kv_s_kv = kv_cache->stride(1);
    kv_s_token = kv_cache->stride(2);
    kv_s_head = kv_cache->stride(3);
    effective_index_slot_mapping = index_slot_mapping.has_value()
                                       ? &index_slot_mapping.value()
                                       : &slot_mapping.value();
  }
  if (q_out.has_value()) {
    TORCH_CHECK(q_out->is_cuda() && q_out->is_contiguous() &&
                    q_out->scalar_type() == qkv.scalar_type() &&
                    q_out->numel() ==
                        static_cast<int64_t>(num_tokens) * nq * kHeadDim,
                "q_out must be a contiguous CUDA tensor matching qkv dtype "
                "with num_tokens * num_heads * 128 elements");
  }
  if (index_q_out.has_value()) {
    TORCH_CHECK(has_index && index_q_out->is_cuda() &&
                    index_q_out->is_contiguous() &&
                    (index_q_out->scalar_type() == qkv.scalar_type() ||
                     index_q_out->scalar_type() ==
                         at::ScalarType::Float8_e4m3fn) &&
                    index_q_out->numel() ==
                        static_cast<int64_t>(num_tokens) * niq * kHeadDim,
                "index_q_out must be contiguous CUDA, qkv dtype or fp8 e4m3, "
                "num_tokens * num_index_heads * 128 elements");
  }

  auto const kFp8 = at::ScalarType::Float8_e4m3fn;
  bool const fp8_idx =
      (index_cache.has_value() && index_cache->scalar_type() == kFp8) ||
      (index_q_out.has_value() && index_q_out->scalar_type() == kFp8);
  if (fp8_idx) {
    TORCH_CHECK(!index_cache.has_value() ||
                    index_cache->scalar_type() == kFp8,
                "fp8 index path: index_cache must be fp8 e4m3");
    TORCH_CHECK(!index_q_out.has_value() ||
                    index_q_out->scalar_type() == kFp8,
                "fp8 index path: index_q_out must be fp8 e4m3");
  }

  const c10::cuda::CUDAGuard device_guard(qkv.get_device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

  auto index_q_out_ptr =
      index_q_out.has_value() ? index_q_out->data_ptr() : nullptr;
  auto index_cache_ptr =
      (insert_kv && has_index && index_cache.has_value())
          ? index_cache->data_ptr()
          : nullptr;

  if (qkv.scalar_type() == at::ScalarType::BFloat16) {
    using st = __nv_bfloat16;
    launchFusedMiniMaxM3<st>(
        reinterpret_cast<st*>(qkv.data_ptr()),
        q_out.has_value()
            ? reinterpret_cast<st*>(q_out->data_ptr())
            : nullptr,
        index_q_out_ptr,
        reinterpret_cast<st const*>(q_norm_weight.data_ptr()),
        reinterpret_cast<st const*>(k_norm_weight.data_ptr()),
        has_index
            ? reinterpret_cast<st const*>(index_q_norm_weight->data_ptr())
            : nullptr,
        has_index
            ? reinterpret_cast<st const*>(index_k_norm_weight->data_ptr())
            : nullptr,
        reinterpret_cast<st const*>(cos_sin_cache.data_ptr()),
        positions.data_ptr<int64_t>(),
        insert_kv ? slot_mapping->data_ptr<int64_t>() : nullptr,
        insert_kv ? effective_index_slot_mapping->data_ptr<int64_t>()
                  : nullptr,
        insert_kv ? reinterpret_cast<st*>(kv_cache->data_ptr()) : nullptr,
        index_cache_ptr, static_cast<float>(eps),
        static_cast<int>(rotary_dim), num_tokens, nq, nkv, niq,
        static_cast<int>(block_size), kv_s_block, kv_s_kv, kv_s_token,
        kv_s_head, has_index, insert_kv, fp8_idx, stream);
  } else {
    using st = __half;
    launchFusedMiniMaxM3<st>(
        reinterpret_cast<st*>(qkv.data_ptr()),
        q_out.has_value()
            ? reinterpret_cast<st*>(q_out->data_ptr())
            : nullptr,
        index_q_out_ptr,
        reinterpret_cast<st const*>(q_norm_weight.data_ptr()),
        reinterpret_cast<st const*>(k_norm_weight.data_ptr()),
        has_index
            ? reinterpret_cast<st const*>(index_q_norm_weight->data_ptr())
            : nullptr,
        has_index
            ? reinterpret_cast<st const*>(index_k_norm_weight->data_ptr())
            : nullptr,
        reinterpret_cast<st const*>(cos_sin_cache.data_ptr()),
        positions.data_ptr<int64_t>(),
        insert_kv ? slot_mapping->data_ptr<int64_t>() : nullptr,
        insert_kv ? effective_index_slot_mapping->data_ptr<int64_t>()
                  : nullptr,
        insert_kv ? reinterpret_cast<st*>(kv_cache->data_ptr()) : nullptr,
        index_cache_ptr, static_cast<float>(eps),
        static_cast<int>(rotary_dim), num_tokens, nq, nkv, niq,
        static_cast<int>(block_size), kv_s_block, kv_s_kv, kv_s_token,
        kv_s_head, has_index, insert_kv, fp8_idx, stream);
  }
}

}  // namespace m3_fused
}  // namespace vllm

TORCH_LIBRARY(_C, m) {
  m.def(
      "fused_minimax_m3_qknorm_rope_kv_insert("
      "Tensor! qkv, Tensor q_norm_weight, Tensor k_norm_weight, "
      "Tensor cos_sin_cache, Tensor positions, int num_heads, "
      "int num_kv_heads, int rotary_dim, float eps, "
      "Tensor? index_q_norm_weight, Tensor? index_k_norm_weight, "
      "int num_index_heads, "
      "Tensor? slot_mapping, Tensor? index_slot_mapping, "
      "Tensor!? kv_cache, Tensor!? index_cache, "
      "int block_size, Tensor!? q_out, Tensor!? index_q_out, "
      "str kv_cache_dtype) -> ()",
      TORCH_FN(vllm::m3_fused::fused_minimax_m3_qknorm_rope_kv_insert));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // The TORCH_LIBRARY block above registers the op at static-init time;
  // importing this module is enough to activate the native kernel.
  m.doc() = "MiniMax-M3 fused qknorm+rope+kv-insert native kernel (PPU)";
}
