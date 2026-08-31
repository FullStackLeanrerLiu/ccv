/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <cute/arch/mma_sm75.hpp>

#include <cute/atom/mma_traits.hpp>
#include <cute/layout.hpp>

namespace cute
{

template <>
struct MMA_Traits<SM75_16x8x8_F32F16F16F32_TN>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16,_8,_8>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,
                         Stride<Stride<_32,_1>,Stride<_16,_8>>>;
  using BLayout = Layout<Shape <Shape < _4,_8>,_2>,
                         Stride<Stride<_16,_1>,_8>>;
  using CLayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,
                         Stride<Stride<_32,_1>,Stride<_16,_8>>>;
};

///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM75_8x8x16_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_8,_8,_16>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,_4>,
                         Stride<Stride<_32,_1>,_8>>;
  using BLayout = Layout<Shape <Shape < _4,_8>,_4>,
                         Stride<Stride<_32,_1>,_8>>;
  using CLayout = Layout<Shape <Shape < _4,_8>,_2>,
                         Stride<Stride<_16,_1>,_8>>;
};

///////////////////////////////////////////////////////////////////////////////

// MMA m8n8k32 TN: Turing native INT4 tensor-core MMA
// (mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32).
//
// The 4-bit atom shares the C/D fragment structure of the 8-bit m8n8k16 atom
// (two int32 C registers), but each uint32 A/B register now packs *8* signed
// 4-bit elements (K = 32) instead of 4 int8 (K = 16), so the A/B fragment
// carries 8 int4b per thread.  The thread / value layout below therefore
// mirrors the INT8 m8n8k16 atom but with the value mode scaled to _8 (cosize
// 256 instead of 128) and the K stride halved (_4 vs _8) so the 8 values a
// thread owns are bit-contiguous 4-bit nibbles along K.  CLayout is unchanged.
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM75_8x8x32_S32S4S4S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int4b_t;
  using ValTypeB = int4b_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_8,_8,_32>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,_8>,
                         Stride<Stride<_32,_1>,_4>>;
  using BLayout = Layout<Shape <Shape < _4,_8>,_8>,
                         Stride<Stride<_32,_1>,_4>>;
  using CLayout = Layout<Shape <Shape < _4,_8>,_2>,
                         Stride<Stride<_16,_1>,_8>>;
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cute
