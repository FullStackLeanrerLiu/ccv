/******************************************************************************
 * Copyright (c) 2026, ccv.
 *
 * SageAttention-style INT8-QK + FP16-PV fused flash-attention forward traits
 * for Turing (sm75 / RTX 2080 Ti).
 *
 * Rationale
 * ---------
 * The original SM75 flash-attention path computes the QK score tile with the
 * FP16 tensor core (SM75_16x8x8_F32F16F16F32_TN) and keeps acc_s (scores) and
 * acc_o (output) in the M16N8K8 C-fragment layout.  That atom is M16N8K8.
 *
 * Turing's INT8 tensor core is a *different* instruction: mma.sync.aligned
 * .m8n8k16.row.col.s32.s8.s8.s32 (SM75_8x8x16_S32S8S8S32_TN, M8N8K16).  Its
 * C/D accumulator is int32 and its per-warp M coverage is 8 (half of the FP16
 * atom's 16).  Consequently the int8 QK accumulator cannot share the FP16
 * C-fragment layout, so we use two *independent* TiledMma instances:
 *   - TiledMmaQk : INT8 mma that computes S = Q8 @ K8^T into an int32 acc_s.
 *   - TiledMmaPv / TiledMma : the original FP16 mma that computes O = P @ V
 *     into the FP16 acc_o fragment.
 * The dequantized fp32 scores are first spilled to a small row-major shared
 * memory tile (sR) and then re-read into the FP16 C-fragment (acc_s) so the
 * existing softmax (softmax_rescale_o / mask) and the FP16 PV stage can be
 * reused verbatim.
 *
 * Note on shared memory: Turing allows at most ~64KB per block.  The FP16 base
 * tile already uses ~16KB-48KB for Q/K/V.  The int8 Q/K buffers are only 1
 * byte/elt (half the fp16 size) and the fp32 score spill tile is kBlockM*kBlockN
 * floats, so we deliberately keep kBlockN small (32/64) for the int8 launches.
 ******************************************************************************/

#pragma once

#include "kernel_traits.h"

#include <cute/algorithm/copy.hpp>
#include <cutlass/cutlass.h>

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type = cutlass::half_t,
         typename Base_ = Flash_fwd_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, false, false, elem_type>>
struct Flash_int8_kernel_traits : public Base_ {

    // Expose the FP16 base traits both as a member typedef (= Base_) and as the
    // injected base-class name, so callers can write Kernel_traits::Base::kSmemSize.
    using Base = Base_;

    // ---- Inherit the FP16 storage / accumulator types and the FP16 PV mma. ----
    using Element = typename Base::Element;            // fp16 storage for Q/K/V/O
    using ElementAccum = typename Base::ElementAccum;  // float accumulator
    using index_t = typename Base::index_t;

    static constexpr int kBlockM = Base::kBlockM;
    static constexpr int kBlockN = Base::kBlockN;
    static constexpr int kHeadDim = Base::kHeadDim;
    static constexpr int kNWarps = Base::kNWarps;
    static constexpr int kNThreads = Base::kNThreads;

    // Original FP16 mma, exposed both under its own name (used by the int8
    // kernel to compute PV) and under the SFINA-ed alias `TiledMma` so that the
    // unchanged softmax / PV / epilogue code keeps compiling verbatim.
    using TiledMmaPv = typename Base::TiledMma;
    using TiledMma = typename Base::TiledMma;

    // ---- INT8-QK stage (Turing mma.s8.m8n8k16). ----
    using QKElem = int8_t;
    using QKAcc = int32_t;

    // S = Q8 @ K8^T  =>  A = Q tile (M = kBlockM rows of Q), B = K tile
    // (M = kBlockN rows of K), C = S.  Warps are laid out along M exactly like
    // the FP16 path so each warp covers a disjoint (kBlockM/kNWarps) x (kBlockN)
    // contiguous slab of S; cute::partition_fragment_C iterates over kBlockM /
    // kBlockN when the tile is smaller than the requested shape.
    // Tile M = 8 (atom M) * kNWarps, Tile N = 16 (2 N8 atoms), Tile K = 16.
    using TiledMmaQk = TiledMMA<
        MMA_Atom<SM75_8x8x16_S32S8S8S32_TN>,
        Layout<Shape<Int<kNWarps>, _1, _1>>,
        Tile<Int<8 * kNWarps>, _16, _16>>;

    // ---- INT8 Q/K shared memory (plain row-major, 1 byte / element). ----
    using Q8SmemLayout = Layout<Shape<Int<kBlockM>, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>;
    using K8SmemLayout = Layout<Shape<Int<kBlockN>, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>;

    // ---- fp32 score spill tile (row-major) + per-row INT8 scales. ----
    // ScoresS is (kBlockM x kBlockN) fp32; Q coefficients are stored scale_q[r]
    // for each query row, K coefficients scale_k[r] for each key row.
    using ScoresSmemLayout = Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<Int<kBlockN>, _1>>;

    static constexpr int kSmemQ8Size = size(Q8SmemLayout{}) * sizeof(QKElem);   // bytes
    static constexpr int kSmemK8Size = size(K8SmemLayout{}) * sizeof(QKElem);   // bytes
    static constexpr int kSmemScaleSize = (kBlockM + kBlockN) * sizeof(float);  // bytes
    static constexpr int kSmemScoresSize = size(ScoresSmemLayout{}) * sizeof(float); // bytes
    static constexpr int kSmemSize = Base::kSmemSize
        + kSmemQ8Size + kSmemK8Size + kSmemScaleSize + kSmemScoresSize;
};

////////////////////////////////////////////////////////////////////////////////////////////////////