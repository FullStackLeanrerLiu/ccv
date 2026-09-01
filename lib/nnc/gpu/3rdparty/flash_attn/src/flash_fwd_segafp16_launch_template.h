/******************************************************************************
 * Copyright (c) 2026, ccv.
 *
 * fp16-SageAttention launch template (Turing sm75).
 *
 * This is the "fp16-SageAttention-style" tuning variant of the standard FP16
 * fused flash-attention kernel.  It reuses the SAME production FP16 fused
 * kernel flash_fwd_kernel / flash_fwd_launch_template.h (QK and PV both on the
 * FP16 tensor core, SM75_16x8x8 / m16n8k8) — no INT8/INT4 quantization — but
 * launches it with a SageAttention-flavored block shape (square-ish tiles with
 * kBlockM == 64, half the default 128-row Q tile) so the two paths can be
 * A/B-compared for the small attention tensors (D=128, R=32, C=32) produced by
 * Z-Image inference on 2080 Ti.
 *
 * Enable via: CCV_SEGA_MODE=fp16  ->  params.is_sega_fp16 = true.
 *
 * PV-stage correctness note: on Turing the FP16 tensor core is m16n8k8 only
 * (there is no FP16 m8n8k4 path); QK/PV both use SM75_16x8x8.  This is a pure
 * FP16 path — it never touches the INT8 m8n8k16 instruction.
 ******************************************************************************/

#pragma once

#include "static_switch.h"
#include "flash.h"
#include "flash_fwd_launch_template.h"

// The fp16-SageAttention variant for Turing uses Q tiles of 64 rows (Sage-like
// square tiles) with NV = 64 for head dim <= 128 to keep smem within Turing's
// ~64KB cap while maximizing the re-usable N slice.
template<typename T>
void run_mha_fwd_segafp16_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        });
    });
}

template<typename T>
void run_mha_fwd_segafp16_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        });
    });
}

template<typename T>
void run_mha_fwd_segafp16_hdim256(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    // Turing caps smem at ~64KB/block (64 x 32 x 256 = 48KB for Q + KV), so use
    // a N=32 tile; a 64x64 tile would exceed the 64KB cap here.
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        });
    });
}

// Dispatcher used by the *_segafp16_sm75.cu translation units.
template<typename T, int Headdim>
void run_mha_fwd_segafp16_(Flash_fwd_params &params, cudaStream_t stream) {
    if constexpr (Headdim == 64) {
        run_mha_fwd_segafp16_hdim64<T>(params, stream);
    } else if constexpr (Headdim == 128) {
        run_mha_fwd_segafp16_hdim128<T>(params, stream);
    } else if constexpr (Headdim == 256) {
        run_mha_fwd_segafp16_hdim256<T>(params, stream);
    } else {
        // Only 64 / 128 / 256 are instantiated for the fp16-SageAttention path.
        run_mha_fwd_<T, Headdim>(params, stream);
    }
}