// Copyright (c) 2026, ccv.
// Splitting the different head dimensions to different files to speed up compilation.
// fp16-SageAttention tuning variant instantiations (Turing sm75).

#include "flash_fwd_segafp16_launch_template.h"

template void run_mha_fwd_segafp16_<cutlass::half_t, 128>(Flash_fwd_params &params, cudaStream_t stream);