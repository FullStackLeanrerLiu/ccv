# Copied from Driss Guessous's PR in PyTorch: https://github.com/pytorch/pytorch/pull/105602

# This file is run to generate the kernel instantiations for the flash_attn kernels
# They are written to several files in order to speed up compilation

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

DTYPE_MAP = {
    "fp16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
}

SM = [80]  # Sm80 kernels support up to
HEAD_DIMENSIONS = [32, 64, 96, 128, 160, 192, 224, 256]
KERNEL_IMPL_TEMPLATE_FWD = """#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_<{DTYPE}, {HEAD_DIM}>(Flash_fwd_params &params, cudaStream_t stream) {{
    run_mha_fwd_hdim{HEAD_DIM}<{DTYPE}>(params, stream);
}}
"""

KERNEL_IMPL_TEMPLATE_FWD_SPLIT = """#include "flash_fwd_launch_template.h"

template void run_mha_fwd_splitkv_dispatch<{DTYPE}, {HEAD_DIM}>(Flash_fwd_params &params, cudaStream_t stream);
"""

# SageAttention-style INT8-QK forward kernel (Turing sm75).  Only the fwd
# direction is produced; there is no int8 split-kv / backward kernel.
INT8_KERNEL_IMPL_TEMPLATE_FWD = """#include "flash_fwd_int8_launch_template.h"

template void run_mha_fwd_int8_<{DTYPE}, {HEAD_DIM}>(Flash_fwd_params &params, cudaStream_t stream);
"""

# SageAttention-style INT4-QK forward kernel (Turing sm75).  Only the fwd
# direction is produced; there is no int4 split-kv / backward kernel.
INT4_KERNEL_IMPL_TEMPLATE_FWD = """#include "flash_fwd_int4_launch_template.h"

template void run_mha_fwd_int4_<{DTYPE}, {HEAD_DIM}>(Flash_fwd_params &params, cudaStream_t stream);
"""

# Head dimensions for which the int8-QK / int4-QK paths are available (sm75 / Turing).
INT8_HEAD_DIMENSIONS = [64, 128, 256]
INT4_HEAD_DIMENSIONS = [64, 128, 256]

KERNEL_IMPL_TEMPLATE_BWD = """#include "flash_bwd_launch_template.h"

template<>
void run_mha_bwd_<{DTYPE}, {HEAD_DIM}>(Flash_bwd_params &params, cudaStream_t stream) {{
    run_mha_bwd_hdim{HEAD_DIM}<{DTYPE}>(params, stream);
}}
"""


@dataclass
class Kernel:
    sm: int
    dtype: str
    head_dim: int
    direction: str
    is_int8: bool = False
    is_int4: bool = False

    @property
    def template(self) -> str:
        if self.is_int4:
            return INT4_KERNEL_IMPL_TEMPLATE_FWD.format(
                DTYPE=DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim
            )
        if self.is_int8:
            return INT8_KERNEL_IMPL_TEMPLATE_FWD.format(
                DTYPE=DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim
            )
        if self.direction == "fwd":
            return KERNEL_IMPL_TEMPLATE_FWD.format(
                DTYPE=DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim
            )
        elif self.direction == "bwd":
            return KERNEL_IMPL_TEMPLATE_BWD.format(
                DTYPE=DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim
            )
        else:
            return KERNEL_IMPL_TEMPLATE_FWD_SPLIT.format(
                DTYPE=DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim
            )

    @property
    def filename(self) -> str:
        if self.is_int8:
            return f"flash_fwd_int8_hdim{self.head_dim}_sm{self.sm}.cu"
        if self.is_int4:
            return f"flash_fwd_int4_hdim{self.head_dim}_sm{self.sm}.cu"
        return f"flash_{self.direction}_hdim{self.head_dim}_{self.dtype}_sm{self.sm}.cu"


def get_all_kernels() -> List[Kernel]:
    # sm80 kernels: all dtypes, all directions (fwd / bwd / fwd_split)
    for dtype, head_dim, sm in itertools.product(DTYPE_MAP.keys(), HEAD_DIMENSIONS, SM):
        for direction in ["fwd", "bwd", "fwd_split"]:
            yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim, direction=direction)
    # sm75 kernels: Turing only has FP16 tensor-core MMA (SM75_16x8x8) and no cp.async,
    # so we emit only FP16 fwd + fwd_split (inference path). bf16 and bwd stay sm80-only.
    for head_dim in HEAD_DIMENSIONS:
        for direction in ["fwd", "fwd_split"]:
            yield Kernel(sm=75, dtype="fp16", head_dim=head_dim, direction=direction)
    # sm75 INT8-QK forward kernels (SageAttention-style w8a8), fp16 inputs only.
    for head_dim in INT8_HEAD_DIMENSIONS:
        yield Kernel(sm=75, dtype="fp16", head_dim=head_dim, direction="fwd", is_int8=True)
    # sm75 INT4-QK forward kernels (SageAttention-style, native INT4 tensor core),
    # fp16 inputs only.
    for head_dim in INT4_HEAD_DIMENSIONS:
        yield Kernel(sm=75, dtype="fp16", head_dim=head_dim, direction="fwd", is_int4=True)


def write_kernel(kernel: Kernel, autogen_dir: Path) -> None:
    prelude = """// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"\n
"""
    (autogen_dir / kernel.filename).write_text(prelude + kernel.template)


def main(output_dir: Optional[str]) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    for kernel in get_all_kernels():
        write_kernel(kernel, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_kernels",
        description="Generate the flash_attention kernels template instantiations",
    )
    # Set an optional output directory
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="Where to generate the kernels "
        " will default to the current directory ",
    )
    args = parser.parse_args()
    main(args.output_dir)
