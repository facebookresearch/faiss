/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// This per-level TU is compiled only with the RVV flags, so it has no
// whole-file guard. The Hamming top-k specialization for RISCV_RVV lives in
// a per-level header, which only this TU includes, before hamming_impl.h
// instantiates the bucket kernel.

#define THE_SIMD_LEVEL SIMDLevel::RISCV_RVV
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/hamming_distance/hamming_computer-rvv.h>
#include <faiss/utils/approx_topk_hamming/approx_topk_hamming-rvv.h>
#include <faiss/utils/hamming_distance/hamming_impl.h>
