/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/// Represents the mode of use of approximate top-k computations
/// that allows to trade accuracy vs speed. So, every options
/// besides EXACT_TOPK increases the speed.
///
/// B represents the number of buckets.
/// D is the number of min-k elements to track within every bucket.
///
/// Default option is EXACT_TOPK.
/// APPROX_TOPK_BUCKETS_B16_D2 is worth starting from, if you'd like
/// to experiment a bit.
///
/// It seems that only the limited number of combinations are
/// meaningful, because of the limited supply of SIMD registers.
/// Also, certain combinations, such as B32_D1 and B16_D1, were concluded
/// to be not very precise in benchmarks, so ones were not introduced.
///
/// TODO: Consider d-ary SIMD heap.

enum ApproxTopK_mode_t : int {
    EXACT_TOPK = 0,
    APPROX_TOPK_BUCKETS_B32_D2 = 1,
    APPROX_TOPK_BUCKETS_B8_D3 = 2,
    APPROX_TOPK_BUCKETS_B16_D2 = 3,
    APPROX_TOPK_BUCKETS_B8_D2 = 4,
};
