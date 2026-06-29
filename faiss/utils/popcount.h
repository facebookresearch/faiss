/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

// popcount = count number of 1s in the binary representation of a number

#ifdef _MSC_VER
#include <intrin.h>
inline int popcount32(uint32_t x) {
    return __popcnt(x);
}
inline int popcount64(uint64_t x) {
    return static_cast<int>(__popcnt64(x));
}
#else
inline int popcount32(uint32_t x) {
    return __builtin_popcount(x);
}
inline int popcount64(uint64_t x) {
    return __builtin_popcountll(x);
}
#endif
