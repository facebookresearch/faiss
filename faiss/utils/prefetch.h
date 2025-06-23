/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// prefetches

#ifdef __AVX__

// AVX

#include <xmmintrin.h>

inline void prefetch_L1(const void* address) {
    _mm_prefetch((const char*)address, _MM_HINT_T0);
}
inline void prefetch_L2(const void* address) {
    _mm_prefetch((const char*)address, _MM_HINT_T1);
}
inline void prefetch_L3(const void* address) {
    _mm_prefetch((const char*)address, _MM_HINT_T2);
}

#elif defined(__aarch64__)

// ARM64

#ifdef _MSC_VER

// todo: arm on MSVC
inline void prefetch_L1(const void* address) {}
inline void prefetch_L2(const void* address) {}
inline void prefetch_L3(const void* address) {}

#else
// arm on non-MSVC

inline void prefetch_L1(const void* address) {
    __builtin_prefetch(address, 0, 3);
}
inline void prefetch_L2(const void* address) {
    __builtin_prefetch(address, 0, 2);
}
inline void prefetch_L3(const void* address) {
    __builtin_prefetch(address, 0, 1);
}
#endif

#else

// a generic platform

#ifdef _MSC_VER

inline void prefetch_L1(const void* address) {}
inline void prefetch_L2(const void* address) {}
inline void prefetch_L3(const void* address) {}

#else

inline void prefetch_L1(const void* address) {
    __builtin_prefetch(address, 0, 3);
}
inline void prefetch_L2(const void* address) {
    __builtin_prefetch(address, 0, 2);
}
inline void prefetch_L3(const void* address) {
    __builtin_prefetch(address, 0, 1);
}

#endif

#endif
