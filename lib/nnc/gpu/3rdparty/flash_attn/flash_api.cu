/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <cutlass/numeric_types.h>

#include "src/flash.h"
#include "src/static_switch.h"

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
#ifdef HAVE_CUDA_SM75
    // sm75 (Turing) builds have FP16 fwd + fwd_split kernels only: bf16 MMA is
    // Ampere-only, so we instantiate the half_t path directly without FP16_SWITCH.
    //
    // Dispatch order on Turing:
    //   1. INT4-QK / INT8-QK quantized paths — EXPERIMENTAL, only when explicitly
    //      requested via CCV_QK_MODE=int4 / int8 (params.is_int4qk / is_int8qk).
    //      They use Turing's native INT4 (m8n8k32) / INT8 (m8n8k16) tensor cores
    //      for QK and the FP16 tensor core (m16n8k8) for PV.
    //   2. fp16-SageAttention tuning variant (CCV_SEGA_MODE=fp16) — FP16 QK + FP16
    //      PV fusion with a SageAttention-flavored tile config for A/B.
    //   3. Standard FP16 fused path (default).
    if (!force_split_kernel && params.num_splits <= 1 &&
        (params.is_int4qk || params.is_int8qk)) {
        const bool use_int4 = params.is_int4qk;
        if (params.d <= 64) {
            if (use_int4) { run_mha_fwd_int4_<cutlass::half_t, 64>(params, stream); }
            else          { run_mha_fwd_int8_<cutlass::half_t, 64>(params, stream); }
        } else if (params.d <= 128) {
            if (use_int4) { run_mha_fwd_int4_<cutlass::half_t, 128>(params, stream); }
            else          { run_mha_fwd_int8_<cutlass::half_t, 128>(params, stream); }
        } else if (params.d <= 256) {
            if (use_int4) { run_mha_fwd_int4_<cutlass::half_t, 256>(params, stream); }
            else          { run_mha_fwd_int8_<cutlass::half_t, 256>(params, stream); }
        } else {
            HEADDIM_SWITCH(params.d, [&] {
                run_mha_fwd_<cutlass::half_t, kHeadDim>(params, stream);
            });
        }
    } else if (!force_split_kernel && params.num_splits <= 1 && params.is_sega_fp16) {
        if (params.d <= 64) {
            run_mha_fwd_segafp16_<cutlass::half_t, 64>(params, stream);
        } else if (params.d <= 128) {
            run_mha_fwd_segafp16_<cutlass::half_t, 128>(params, stream);
        } else if (params.d <= 256) {
            run_mha_fwd_segafp16_<cutlass::half_t, 256>(params, stream);
        } else {
            HEADDIM_SWITCH(params.d, [&] {
                run_mha_fwd_<cutlass::half_t, kHeadDim>(params, stream);
            });
        }
    } else {
        HEADDIM_SWITCH(params.d, [&] {
            if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
                run_mha_fwd_<cutlass::half_t, kHeadDim>(params, stream);
            } else {
                run_mha_fwd_splitkv_dispatch<cutlass::half_t, kHeadDim>(params, stream);
            }
        });
    }
#else
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
                run_mha_fwd_<elem_type, kHeadDim>(params, stream);
            } else {
                run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim>(params, stream);
            }
        });
    });
#endif
}

// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

#ifdef HAVE_CUDA_SM80
void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            run_mha_bwd_<elem_type, kHeadDim>(params, stream);
        });
    });
}
#endif
