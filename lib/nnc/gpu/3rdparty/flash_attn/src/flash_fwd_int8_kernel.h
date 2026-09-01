/******************************************************************************
 * Copyright (c) 2026, ccv.
 *
 * SageAttention-style INT8-QK + FP16-PV flash-attention forward kernel
 * (Turing sm75 only).  Q/K are dynamically quantized to int8 with per-block
 * scales (one scale per Q tile and one per K tile — the same granularity as
 * the community SageAttention per_block_int8 sm75 path, BLK = kBlockM /
 * kBlockN x D).  The QK score GEMM runs on Turing's INT8 tensor core
 * (mma.s8.m8n8k16) into an int32 accumulator, the result is dequantized
 * (x scale_q * scale_k) to fp32 and spilled through a row-major shared-memory
 * tile, then re-read into the original FP16 C-fragment (`acc_s`) so the
 * unchanged FP16 softmax and FP16 PV stage can be reused verbatim.
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

#include "block_info.h"
#include "kernel_traits.h"
#include "flash_int8_kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"
#include "dropout.h"
#include "rotary.h"

namespace flash {

using namespace cute;

// Range used by the symmetric (zero-point = 0) INT8 quantizer.
constexpr float kInt8Range = 127.f;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Dynamic per-block INT8 quantization of a tile in shared-memory.
//
// One scale is computed for the whole (nRows x HeadDim) tile: a block-wide
// absmax reduced over all kNThreads threads (warp shuffles + a 1-float smem
// fold), then scale = absmax / 127 and Q = round(fp16 * 127 / absmax) clamped
// to [-128, 127] are written.  This mirrors the per_block_int8 granularity of
// the community SageAttention sm75 path (one scale per BLK x D block, with
// BLK = kBlockM / kBlockN here) and removes the per-row scale array + per-row
// divisions of the previous implementation.
//
// `scale_out` points at a 2-float smem slot: [0] = scale (absmax/127, or 1.0
// for a silent tile), [1] = recip (127/absmax, 0 for a silent tile).  The
// scale pair is written by thread 0 after a block reduction; the trailing
// __syncthreads() makes it visible to all threads before conversion.
//
// `s` may be any (possibly swizzled) FP16 smem tensor; only `s8` is assumed
// plain row-major.
////////////////////////////////////////////////////////////////////////////////////////////////////
template<int nRows, int HeadDim, int kNThreads, typename F16Tensor>
__device__ __forceinline__ void int8_quantize_block(F16Tensor const& s, int8_t* __restrict__ s8,
                                                    float* __restrict__ scale_out) {
    // ---- 1) block-wide absmax over all nRows x HeadDim elements ----
    float local = 0.f;
    #pragma unroll 1
    for (int r = 0; r < nRows; ++r) {
        #pragma unroll 1
        for (int j = threadIdx.x; j < HeadDim; j += kNThreads) {
            const float v = (float)s(r, j);
            local = fmaxf(local, fabsf(v));
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local = fmaxf(local, __shfl_xor_sync(0xffffffffu, local, off));
    __shared__ float warp_amax[8];  // kNThreads/32 <= 8 warps for all instantiations
    if ((threadIdx.x & 31) == 0) { warp_amax[threadIdx.x >> 5] = local; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float amax = 0.f;
        #pragma unroll
        for (int w = 0; w < kNThreads / 32; ++w) { amax = fmaxf(amax, warp_amax[w]); }
        const bool has_signal = amax > 1e-12f;
        scale_out[0] = has_signal ? amax / kInt8Range : 1.f;  // scale
        scale_out[1] = has_signal ? kInt8Range / amax : 0.f;  // recip = 127 / absmax
    }
    __syncthreads();

    // ---- 2) convert + store using the broadcast [scale, recip] pair ----
    const float inv = scale_out[1];
    #pragma unroll 1
    for (int r = 0; r < nRows; ++r) {
        int8_t* const row8 = s8 + r * HeadDim;
        #pragma unroll 1
        for (int j = threadIdx.x; j < HeadDim; j += kNThreads) {
            const float v = (float)s(r, j) * inv;
            int vi = (int)(v < 0.f ? v - 0.5f : v + 0.5f);
            vi = vi < -128 ? -128 : (vi > 127 ? 127 : vi);
            row8[j] = (int8_t)vi;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// INT8-QK forward attention for one row-block (int8 score path, FP16 PV path).
////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Return_softmax, typename Params>
inline __device__ void compute_attn_int8_1rowblock(const Params &params, const int bidb, const int bidh, const int m_block) {

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

    // ---- INT8 buffers, allocated right after the FP16 Q/K/V smem region. ----
    typename Kernel_traits::TiledMmaQk tiled_mma_qk;
    auto thr_mma_qk = tiled_mma_qk.get_thread_slice(tidx);

    int8_t* const sQ8ptr = reinterpret_cast<int8_t*>(smem_) + Kernel_traits::Base::kSmemSize;
    int8_t* const sK8ptr = sQ8ptr + Kernel_traits::kSmemQ8Size;
    float*  const sQsc  = reinterpret_cast<float*>(sK8ptr + Kernel_traits::kSmemK8Size);
    float*  const sKsc  = sQsc + 2;   // [scale, recip] per tile (2 floats each)
    float*  const sRptr = sKsc + 2;
    Tensor sQ8 = make_tensor(make_smem_ptr(sQ8ptr), typename Kernel_traits::Q8SmemLayout{});
    Tensor sK8 = make_tensor(make_smem_ptr(sK8ptr), typename Kernel_traits::K8SmemLayout{});
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

    // INT8 QK smem copy atoms.  Use a plain DefaultCopy: the int8 Q/K tiles are
    // simple row-major (no 16-bit LDSM swizzle needed for the m8n8k16 instruction).
    auto smem_tiled_copy_Q8 = make_tiled_copy_A(Copy_Atom<DefaultCopy, typename Kernel_traits::QKElem>{}, tiled_mma_qk);
    auto smem_thr_copy_Q8 = smem_tiled_copy_Q8.get_thread_slice(tidx);
    Tensor tSsQ8 = smem_thr_copy_Q8.partition_S(sQ8);
    auto smem_tiled_copy_K8 = make_tiled_copy_B(Copy_Atom<DefaultCopy, typename Kernel_traits::QKElem>{}, tiled_mma_qk);
    auto smem_thr_copy_K8 = smem_tiled_copy_K8.get_thread_slice(tidx);
    Tensor tSsK8 = smem_thr_copy_K8.partition_S(sK8);

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

    // Quantize Q to int8 once (per-block scale is constant across all K blocks).
    __syncthreads();
    flash::int8_quantize_block<kBlockM, kHeadDim, kNThreads>(sQ, sQ8ptr, sQsc);
    __syncthreads();
    const float sQscale = sQsc[0];   // per-block Q scale (registers after sync)

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

        // ---------------- INT8-QK stage ----------------
        flash::int8_quantize_block<kBlockN, kHeadDim, kNThreads>(sK, sK8ptr, sKsc);
        __syncthreads();
        const float score_scale = sQscale * sKsc[0];   // per-block dequant scale

        Tensor tSrQ8 = thr_mma_qk.partition_fragment_A(sQ8);
        Tensor tSrK8 = thr_mma_qk.partition_fragment_B(sK8);
        Tensor acc_s8 = partition_fragment_C(tiled_mma_qk, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s8);
        flash::gemm(acc_s8, tSrQ8, tSrK8, tSsQ8, tSsK8, tiled_mma_qk,
                    smem_tiled_copy_Q8, smem_tiled_copy_K8, smem_thr_copy_Q8, smem_thr_copy_K8);

        // Dequantize acc_s8 (int32) -> sR (row-major fp32) using the per-block scales.
        Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
        Tensor tScS = thr_mma_qk.partition_C(cS);
        #pragma unroll
        for (int i = 0; i < size(acc_s8); ++i) {
            const int row = get<0>(tScS(i));
            const int col = get<1>(tScS(i));
            sR(row, col) = (float)acc_s8(i) * score_scale;
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

        // ---------------- INT8-QK stage ----------------
        flash::int8_quantize_block<kBlockN, kHeadDim, kNThreads>(sK, sK8ptr, sKsc);
        __syncthreads();
        const float score_scale = sQscale * sKsc[0];   // per-block dequant scale

        Tensor tSrQ8 = thr_mma_qk.partition_fragment_A(sQ8);
        Tensor tSrK8 = thr_mma_qk.partition_fragment_B(sK8);
        Tensor acc_s8 = partition_fragment_C(tiled_mma_qk, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s8);
        flash::gemm(acc_s8, tSrQ8, tSrK8, tSsQ8, tSsK8, tiled_mma_qk,
                    smem_tiled_copy_Q8, smem_tiled_copy_K8, smem_thr_copy_Q8, smem_thr_copy_K8);

        // Dequantize acc_s8 (int32) -> sR (row-major fp32) using the per-block scales.
        Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
        Tensor tScS = thr_mma_qk.partition_C(cS);
        #pragma unroll
        for (int i = 0; i < size(acc_s8); ++i) {
            const int row = get<0>(tScS(i));
            const int col = get<1>(tScS(i));
            sR(row, col) = (float)acc_s8(i) * score_scale;
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
__device__ __forceinline__ void compute_attn_int8(const Params &params) {
    const int m_block = blockIdx.x;
    const int bidh = blockIdx.z;
    const int bidb = blockIdx.y;
    flash::compute_attn_int8_1rowblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Return_softmax>(params, bidb, bidh, m_block);
}

}  // namespace flash

////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel entry point.
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi,
         bool Is_even_MN, bool Is_even_K, bool Return_softmax>
__global__ void flash_fwd_int8_kernel(const Flash_fwd_params params) {
    flash::compute_attn_int8<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Return_softmax>(params);
}