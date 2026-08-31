/******************************************************************************
 * Copyright (c) 2026, ccv.
 *
 * Launch templates for the SageAttention-style INT8-QK (sm75 only) forward
 * path.  Mirrors flash_fwd_launch_template.h but instantiates
 * flash_fwd_int8_kernel with Flash_int8_kernel_traits.
 *
 * Block shapes here are chosen to keep total shared memory (FP16 Q/K/V + int8
 * Q/K + fp32 score spill) within Turing's ~64KB per block limit.  On a real
 * Turing host these should be re-tuned for occupancy/latency.
 ******************************************************************************/

#pragma once

#include "static_switch.h"
#include "flash.h"
#include "flash_int8_kernel_traits.h"
#include "flash_fwd_int8_kernel.h"

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd_int8(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;

    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        auto kernel = &flash_fwd_int8_kernel<Kernel_traits, Is_dropout, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, ReturnSoftmaxConst && Is_dropout>;
                        if (smem_size >= 48 * 1024) {
                            C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                        }
                        kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    });
                });
            });
        });
    });
}

// The int8-QK path is registered for head dimensions 64 / 128 / 256 only.
template<typename T>
void run_mha_fwd_int8_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    constexpr static int kBlockM = 64;
    constexpr static int kBlockN = 64;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            run_flash_fwd_int8<Flash_int8_kernel_traits<Headdim, kBlockM, kBlockN, 4>, Is_dropout, Is_causal>(params, stream);
        });
    });
}

template<typename T>
void run_mha_fwd_int8_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    constexpr static int kBlockM = 64;
    constexpr static int kBlockN = 32;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            run_flash_fwd_int8<Flash_int8_kernel_traits<Headdim, kBlockM, kBlockN, 4>, Is_dropout, Is_causal>(params, stream);
        });
    });
}

template<typename T>
void run_mha_fwd_int8_hdim256(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    constexpr static int kBlockM = 32;
    constexpr static int kBlockN = 16;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            run_flash_fwd_int8<Flash_int8_kernel_traits<Headdim, kBlockM, kBlockN, 4>, Is_dropout, Is_causal>(params, stream);
        });
    });
}

// Dispatcher used by the generated *_int8_sm75.cu translation units.
template<typename T, int Headdim>
void run_mha_fwd_int8_(Flash_fwd_params &params, cudaStream_t stream) {
    if constexpr (Headdim == 64) {
        run_mha_fwd_int8_hdim64<T>(params, stream);
    } else if constexpr (Headdim == 128) {
        run_mha_fwd_int8_hdim128<T>(params, stream);
    } else if constexpr (Headdim == 256) {
        run_mha_fwd_int8_hdim256<T>(params, stream);
    } else {
        // Only 64 / 128 / 256 are instantiated for the int8-QK path.
        (void)params; (void)stream;
        assert(false);
    }
}