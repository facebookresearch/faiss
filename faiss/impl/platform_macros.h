/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once


#ifdef _MSC_VER
// Windows specific macros


#ifdef FAISS_MAIN_LIB
#define FAISS_API __declspec(dllexport)
#else // _FAISS_MAIN_LIB
#define FAISS_API __declspec(dllimport)
#endif // FAISS_MAIN_LIB

#define __PRETTY_FUNCTION__ __FUNCSIG__

#define posix_memalign(p, a, s) (((*(p)) = _aligned_malloc((s), (a))), *(p) ?0 :errno)
#define posix_memalign_free _aligned_free

// This does not seem to work so disable for now.
// Alignment does not matter if the code is compiled with simdlib_emulated
// but it would matter with read AVX.

// #define ALIGNED(x) __declspec(align(x))
#define ALIGNED(x)

#include <intrin.h>

inline int __builtin_ctzll(uint64_t x) {
    unsigned long ret;
    _BitScanForward64(&ret, x);
    return (int)ret;
}

inline int __builtin_ctz(unsigned long x) {
    unsigned long ret;
    _BitScanForward(&ret, x);
    return (int)ret;
}


#else
// Linux and OSX

#define FAISS_API
#define posix_memalign_free free
#define ALIGNED(x) __attribute__ ((aligned(x)))

#endif // _MSC_VER



