/******************************************************************************
 * Copyright (c) 2026, ccv.
 *
 * SageAttention-style INT4-QK + FP16-PV fused flash-attention forward traits
 * for Turing (sm75 / RTX 2080 Ti).
 *
 * Rationale
 * ---------
 * Identical scaffolding to flash_int8_kernel_traits.h: the original FP16 PV
 * mma and C-fragment layout are inherited from the same FP16 Base, and a
 * *separate* INT4 TiledMma ("TiledMmaQk") computes S = Q4 @ K4^T on Turing's
 * native 4-bit tensor core (mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32,
 * SM75_8x8x32_S32S4S4S32_TN, M8 N8 K32) into an int32 accumulator.  The
 * dequantized fp32 scores are spilled through a small row-major shared tile
 * (sR) and re-read into the FP16 C-fragment so the unchanged FP16 softmax and
 * FP16 PV stage are reused verbatim.
 *
 * Shared memory notes
 * -------------------
 * The INT4 Q/K tiles are stored PACKED (2 signed-4-bit values per byte) so
 * that the Q4 tile only uses kBlockM * kHeadDim / 2 bytes (half of the INT8
 * tile).  Per-row fp32 scales use the symmetric INT4 range [-8, 7], i.e.
 * scale = absmax / 8 (with the amax == 0 row guarded to scale 1.0).  Together
 * with the FP16 Q/K/V tiles and the fp32 score spill, total smem stays well
 * within Turing's ~64 KB per-block cap for the launch shapes in
 * flash_fwd_int4_launch_template.h.
 ******************************************************************************/

#pragma once

#include "kernel_traits.h"

#include <cute/algorithm/copy.hpp>
#include <cutlass/cutlass.h>

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type = cutlass::half_t,
         typename Base_ = Flash_fwd_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, false, false, elem_type>>
struct Flash_int4_kernel_traits : public Base_ {

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

    // Original FP16 mma, exposed both under its own name (used by the int4
    // kernel to compute PV) and under the alias `TiledMma` so that the
    // unchanged softmax / PV / epilogue code keeps compiling verbatim.
    using TiledMmaPv = typename Base::TiledMma;
    using TiledMma = typename Base::TiledMma;

    // ---- INT4-QK stage (Turing mma.s4.m8n8k32). ----
    // Logical QK element is cutlass::int4b_t (range -8..7, 4-bit).  In shared
    // memory it is stored packed as raw bytes (2 int4 per byte) and unpacked
    // into uint32 MMA registers immediately before the MMA (see the kernel).
    using QKElem = cutlass::int4b_t;
    using QKAcc = int32_t;

    // S = Q4 @ K4^T =>  A = Q tile (M = kBlockM), B = K tile (M = kBlockN).
    // Warps are laid out along M exactly like the FP16 path; every warp covers
    // a disjoint (kBlockM/kNWarps) x (kBlockN) slab of S.
    // Tile M = 8 (atom M) * kNWarps, Tile N = 16 (2 N8 atoms), Tile K = 32
    // (one full m8n8k32 atom covers 32 dims of the head dimension).
    using TiledMmaQk = TiledMMA<
        MMA_Atom<SM75_8x8x32_S32S4S4S32_TN>,
        Layout<Shape<Int<kNWarps>, _1, _1>>,
        Tile<Int<8 * kNWarps>, _16, _32>>;

    // ---- INT4 Q/K shared memory (packed row-major bytes, 2 int4 per byte). ----
    // The second dimension is kHeadDim / 2 because every byte caches two 4-bit
    // values; stride kHeadDim/2 is the packed row length.  The intrinsic layout
    // maps element (row, col) of the *logical* [kBlockN/M, kHeadDim] int4 tile
    // to byte (col >> 1) with the low nibble carrying even k, high nibble odd k.
    using Q4SmemLayout = Layout<Shape<Int<kBlockM>, Int<kHeadDim / 2>>, Stride<Int<kHeadDim / 2>, _1>>;
    using K4SmemLayout = Layout<Shape<Int<kBlockN>, Int<kHeadDim / 2>>, Stride<Int<kHeadDim / 2>, _1>>;

    // ---- fp32 score spill tile (row-major) + per-row INT4 scales. ----
    // ScoresS is (kBlockM x kBlockN) fp32; Q coefficients are stored scale_q[r]
    // for each query row, K coefficients scale_k[r] for each key row.
    using ScoresSmemLayout = Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<Int<kBlockN>, _1>>;

    static constexpr int kSmemQ4Size = kBlockM * (kHeadDim / 2);   // bytes (packed, 2int4/byte)
    static constexpr int kSmemK4Size = kBlockN * (kHeadDim / 2);   // bytes (packed, 2int4/byte)
    static constexpr int kSmemScaleSize = (kBlockM + kBlockN) * sizeof(float);  // bytes
    static constexpr int kSmemScoresSize = size(ScoresSmemLayout{}) * sizeof(float); // bytes
    static constexpr int kSmemSize = Base::kSmemSize
        + kSmemQ4Size + kSmemK4Size + kSmemScaleSize + kSmemScoresSize;
};

////////////////////////////////////////////////////////////////////////////////////////////////////