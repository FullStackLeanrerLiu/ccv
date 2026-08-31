/******************************************************************************
 * Copyright (c) 2026, ccv.
 *
 * SageAttention-style INT4-QK + FP16-PV flash-attention forward kernel
 * (Turing sm75 only).  Q/K are dynamically quantized to signed 4-bit (int4),
 * stored in shared memory PACKED two-values-per-byte, the QK score GEMM runs
 * on Turing's native INT4 tensor core (mma.s4.m8n8k32) into an int32
 * accumulator, the result is dequantized (x scale_q * scale_k) to fp32 and
 * spilled through a row-major shared-memory tile, then re-read into the
 * original FP16 C-fragment (`acc_s`) so the unchanged FP16 softmax and FP16
 * PV stage can be reused verbatim.
 *
 * This kernel intentionally does NOT modify flash_fwd_kernel.h: the existing
 * FP16 / FP32 (sm75 / sm80) forward kernels and the backward kernels are left
 * untouched.
 ******************************************************************************/

#pragma once

#include <cute/algorithm/copy.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/integer_subbyte.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "flash_int4_kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"
#include "dropout.h"
#include "rotary.h"

namespace flash {

using namespace cute;

// Half of the full symmetric signed-INT4 range [-8, 7]: absmax / 8.  Using
// absmax / 8 keeps the dequantized score in [-1, 1] before scaling by
// scale_q * scale_k (which together scale back to ~|q||k| like INT8/127^2).
constexpr float kInt4Range = 8.f;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Dynamic per-row INT4 quantization of a tile in shared-memory + 2-per-byte packing.
//
// Each thread owns one entire row (or, when there are fewer rows than threads,
// multiple rows).  For every row it computes the absmax over the full head
// dimension, derives scale = absmax / 8 (guarding the all-zero row to 1.0),
// stores it into `scales` (fp32 shared memory), and writes Q = round(fp16 *
// 8 / absmax) clamped to [-8, 7] into the destination tile, PACKING every two
// consecutive int4 into a single byte (low nibble = even j, high nibble = odd
// j).  `s` is the (swizzled) FP16 tile; `s4` points to the packed byte tile
// whose row length is HeadDim / 2 bytes.
////////////////////////////////////////////////////////////////////////////////////////////////////
template<int nRows, int HeadDim, int kNThreads, typename F16Tensor>
__device__ __forceinline__ void int4_pack_rows(F16Tensor const& s, uint8_t* __restrict__ s4,
                                               float* __restrict__ scales) {
    #pragma unroll 1
    for (int r = threadIdx.x; r < nRows; r += kNThreads) {
        float amax = 0.f;
        #pragma unroll 1
        for (int j = 0; j < HeadDim; ++j) {
            const float v = (float)s(r, j);
            amax = fmaxf(amax, fabsf(v));
        }
        const float scale = amax > 1e-12f ? amax / kInt4Range : 1.f;
        const float inv = amax > 1e-12f ? kInt4Range / amax : 0.f;
        scales[r] = scale;
        uint8_t* const row4 = s4 + r * (HeadDim / 2);
        #pragma unroll 1
        for (int j = 0; j < HeadDim; j += 2) {
            const float v0 = (float)s(r, j + 0) * inv;
            const float v1 = (float)s(r, j + 1) * inv;
            int q0 = (int)(v0 < 0.f ? v0 - 0.5f : v0 + 0.5f);
            int q1 = (int)(v1 < 0.f ? v1 - 0.5f : v1 + 0.5f);
            q0 = q0 < -8 ? -8 : (q0 > 7 ? 7 : q0);
            q1 = q1 < -8 ? -8 : (q1 > 7 ? 7 : q1);
            row4[j / 2] = (uint8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Fill a per-thread INT4 fragment (value type cute::int4b_t) by reading nibbles
// out of the PACKED byte tile.  `tFrag` is the fragment produced by
// `partition_fragment_A/B(...)`; `tCoord` is the matching coordinate fragment
// from partitioning an identity M x K tensor, so element e of the (flattened)
// fragment holds logical (row, k) coordinates.  The m8n8k32 S4 register layout
// is determined by cute from the MMA_Traits; the fragment values are integer
// S4 values in [-8, 7] which cute packs 2-per-byte into the uint32 A/B register
// at MMA time.
////////////////////////////////////////////////////////////////////////////////////////////////////
template<int RowStrideBytes,
         typename ValFrag, typename CoordFrag>
__device__ __forceinline__ void fill_int4_fragment(ValFrag& tFrag, CoordFrag const& tCoord,
                                                   uint8_t const* __restrict__ s4) {
    auto f = flatten(tFrag);
    auto c = flatten(tCoord);
    #pragma unroll
    for (int e = 0; e < size(f); ++e) {
        const int row = get<0>(c(e));
        const int k   = get<1>(c(e));
        const uint8_t byte = s4[row * RowStrideBytes + (k >> 1)];
        int nib = (k & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
        if (nib & 8) { nib -= 16; }  // sign extend 4-bit -> [-8, 7]
        f(e) = cute::int4b_t(nib);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// INT4-QK forward attention for one row-block (int4 score path, FP16 PV path).
////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Return_softmax, typename Params>
inline __device__ void compute_attn_int4_1rowblock(const Params &params, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kHeadDimBytes = kHeadDim / 2;   // packed INT4 row length

    auto seed_offset = philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)
        Tensor mLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr)),
                                  make_shape(params.b, params.h, params.seqlen_q),
                                  make_stride(params.h * params.seqlen_q, params.seqlen_q, _1{}));
        Tensor gLSE = local_tile(mLSE(bidb, bidh, _), Shape<Int<kBlockM>>{}, make_coord(m_block));

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                          + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));
    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                          + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    // ---- INT4 buffers, allocated right after the FP16 Q/K/V smem region. ----
    typename Kernel_traits::TiledMmaQk tiled_mma_qk;
    auto thr_mma_qk = tiled_mma_qk.get_thread_slice(tidx);

    uint8_t* const sQ4ptr = reinterpret_cast<uint8_t*>(smem_) + Kernel_traits::Base::kSmemSize;
    uint8_t* const sK4ptr = sQ4ptr + Kernel_traits::kSmemQ4Size;
    float*  const sQsc  = reinterpret_cast<float*>(sK4ptr + Kernel_traits::kSmemK4Size);
    float*  const sKsc  = sQsc + Kernel_traits::kBlockM;
    float*  const sRptr = sKsc + Kernel_traits::kBlockN;
    Tensor sR  = make_tensor(make_smem_ptr(sRptr),  typename Kernel_traits::ScoresSmemLayout{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    // The original FP16 PV mma (also used for the acc_s / acc_o C-fragment
    // layout and the output epilogue).
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
    Tensor tSgS = thr_mma.partition_C(gP);
    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    // FP16 smem tiled copies (still needed for the PV stage).
    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    // FP32 score re-read into the FP16 C-fragment (acc_s).
    auto smem_tiled_copy_R = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomOaccum{}, tiled_mma);
    auto smem_thr_copy_R = smem_tiled_copy_R.get_thread_slice(tidx);

    // PREDICATES
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                       binfo.actual_seqlen_q - m_block * kBlockM);
    if (!Is_even_K) { __syncthreads(); }

    // Quantize Q to packed int4 once (the Q tile does not change across K blocks).
    __syncthreads();
    flash::int4_pack_rows<kBlockM, kHeadDim, kNThreads>(sQ, sQ4ptr, sQsc);
    __syncthreads();

    int n_block = n_block_max - 1;
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - n_block * kBlockN);
    cute::cp_async_fence();

    clear(acc_o);

    flash::Softmax<2 * size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {

        // fp32 acc_s, laid out as the original FP16 C-fragment (populated below).
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        flash::cp_async_wait<0>();
        __syncthreads();

        // Advance gV
        if (masking_step > 0) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
        } else {
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
            );
        }
        cute::cp_async_fence();

        // ---------------- INT4-QK stage ----------------
        int4_pack_rows<kBlockN, kHeadDim, kNThreads>(sK, sK4ptr, sKsc);
        __syncthreads();

        // Build the per-thread A/B INT4 fragments and fill them from the packed
        // byte tiles (sub-byte values), then run the INT4 mma through cute::gemm
        // exactly like the INT8 kernel's flash::gemm loop.
        Tensor cA = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
        Tensor tAcA = thr_mma_qk.partition_fragment_A(cA);
        Tensor cB = make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
        Tensor tAcB = thr_mma_qk.partition_fragment_B(cB);
        Tensor tSrQ4 = make_fragment_like<cute::int4b_t>(tAcA);
        Tensor tSrK4 = make_fragment_like<cute::int4b_t>(tAcB);
        flash::fill_int4_fragment<kHeadDimBytes>(tSrQ4, tAcA, sQ4ptr);
        flash::fill_int4_fragment<kHeadDimBytes>(tSrK4, tAcB, sK4ptr);

        Tensor acc_s8 = partition_fragment_C(tiled_mma_qk, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s8);
        // Thread-local register GEMM.  Must pass the INT4 MMA_Atom (TiledMMA's
        // base, Kernel_traits::TiledMmaQk::Atom) rather than the TiledMMA itself:
        // cute::gemm's register-geMM overloads only accept an MMA_Atom and do not
        // deduce through the TiledMMA -> MMA_Atom base class.  Pass the *full*
        // rank-3 A/B fragments; layout[5] walks the K-mode internally, mirroring
        // the INT8 kernel's gemm loop.  acc_s8 is used both as the accumulated C
        // and the output D (D = A*B + C).
        cute::gemm(typename Kernel_traits::TiledMmaQk::Atom{}, acc_s8, tSrQ4, tSrK4, acc_s8);

        // Dequantize acc_s8 (int32) -> sR (row-major fp32) using per-row scales.
        Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
        Tensor tScS = thr_mma_qk.partition_C(cS);
        #pragma unroll
        for (int i = 0; i < size(acc_s8); ++i) {
            const int row = get<0>(tScS(i));
            const int col = get<1>(tScS(i));
            sR(row, col) = (float)acc_s8(i) * sQsc[row] * sKsc[col];
        }

        __syncthreads();
        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            cute::cp_async_fence();
        }

        // Re-read the dequantized scores from sR into the FP16 C-fragment acc_s.
        Tensor taccRsR = smem_thr_copy_R.partition_S(sR);
        Tensor taccRs = smem_thr_copy_R.retile_D(acc_s);
        cute::copy(smem_tiled_copy_R, taccRsR, taccRs);

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        // Original fp32 softmax with online rescaling of acc_o.
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16 (P) and run the FP16 PV stage.
        Tensor rP = flash::convert_type<Element>(acc_s);
        int block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
        int block_col_idx = n_block * (kBlockN / 32);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                rP_drop, block_row_idx, block_col_idx, kNWarps
            );
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) { dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps); }

        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // Iterations where we don't need masking on S.
    for (; n_block >= n_block_min; --n_block) {

        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        flash::cp_async_wait<0>();
        __syncthreads();
        flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
        cute::cp_async_fence();

        // ---------------- INT4-QK stage ----------------
        int4_pack_rows<kBlockN, kHeadDim, kNThreads>(sK, sK4ptr, sKsc);
        __syncthreads();

        Tensor cA = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
        Tensor tAcA = thr_mma_qk.partition_fragment_A(cA);
        Tensor cB = make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
        Tensor tAcB = thr_mma_qk.partition_fragment_B(cB);
        Tensor tSrQ4 = make_fragment_like<cute::int4b_t>(tAcA);
        Tensor tSrK4 = make_fragment_like<cute::int4b_t>(tAcB);
        flash::fill_int4_fragment<kHeadDimBytes>(tSrQ4, tAcA, sQ4ptr);
        flash::fill_int4_fragment<kHeadDimBytes>(tSrK4, tAcB, sK4ptr);

        Tensor acc_s8 = partition_fragment_C(tiled_mma_qk, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s8);
        cute::gemm(typename Kernel_traits::TiledMmaQk::Atom{}, acc_s8, tSrQ4, tSrK4, acc_s8);

        Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
        Tensor tScS = thr_mma_qk.partition_C(cS);
        #pragma unroll
        for (int i = 0; i < size(acc_s8); ++i) {
            const int row = get<0>(tScS(i));
            const int col = get<1>(tScS(i));
            sR(row, col) = (float)acc_s8(i) * sQsc[row] * sKsc[col];
        }

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            cute::cp_async_fence();
        }

        Tensor taccRsR = smem_thr_copy_R.partition_S(sR);
        Tensor taccRs = smem_thr_copy_R.retile_D(acc_s);
        cute::copy(smem_tiled_copy_R, taccRsR, taccRs);

        mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);
        int block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
        int block_col_idx = n_block * (kBlockN / 32);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                rP_drop, block_row_idx, block_col_idx, kNWarps
            );
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) { dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps); }

        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue (identical to the original FP16 kernel).
    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    Tensor rO = flash::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                          + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));
    Tensor mLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr)),
                              make_shape(params.b, params.h, params.seqlen_q),
                              make_stride(params.h * params.seqlen_q, params.seqlen_q, _1{}));
    Tensor gLSE = local_tile(mLSE(bidb, bidh, _), Shape<Int<kBlockM>>{}, make_coord(m_block));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor taccOcO = thr_mma.partition_C(caccO);
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));
    if (get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi,
         bool Is_even_MN, bool Is_even_K, bool Return_softmax, typename Params>
__device__ __forceinline__ void compute_attn_int4(const Params &params) {
    const int m_block = blockIdx.x;
    const int bidh = blockIdx.z;
    const int bidb = blockIdx.y;
    flash::compute_attn_int4_1rowblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Return_softmax>(params, bidb, bidh, m_block);
}

}  // namespace flash

////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel entry point.
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi,
         bool Is_even_MN, bool Is_even_K, bool Return_softmax>
__global__ void flash_fwd_int4_kernel(const Flash_fwd_params params) {
    flash::compute_attn_int4<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Return_softmax>(params);
}