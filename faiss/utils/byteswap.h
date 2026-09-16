/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

// byteswap = reverse the byte order of an integer

#ifdef _MSC_VER
#include <cstdlib>
inline uint32_t byteswap32(uint32_t x) {
    return _byteswap_ulong(x);
}
inline uint64_t byteswap64(uint64_t x) {
    return _byteswap_uint64(x);
}
#else
inline uint32_t byteswap32(uint32_t x) {
    return __builtin_bswap32(x);
}
inline uint64_t byteswap64(uint64_t x) {
    return __builtin_bswap64(x);
}
#endif
