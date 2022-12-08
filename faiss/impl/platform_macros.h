/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// basic int types and size_t
#include <cstdint>
#include <cstdio>

#ifdef _MSC_VER

/*******************************************************
 * Windows specific macros
 *******************************************************/

#ifdef FAISS_MAIN_LIB
#define FAISS_API __declspec(dllexport)
#else // _FAISS_MAIN_LIB
#define FAISS_API __declspec(dllimport)
#endif // FAISS_MAIN_LIB

#ifdef _MSC_VER
#define strtok_r strtok_s
#endif // _MSC_VER

#define __PRETTY_FUNCTION__ __FUNCSIG__

#define posix_memalign(p, a, s) \
    (((*(p)) = _aligned_malloc((s), (a))), *(p) ? 0 : errno)
#define posix_memalign_free _aligned_free

// aligned should be in front of the declaration
#define ALIGNED(x) __declspec(align(x))

// redefine the GCC intrinsics with Windows equivalents

#include <intrin.h>

inline int __builtin_ctzll(uint64_t x) {
    unsigned long ret;
    _BitScanForward64(&ret, x);
    return (int)ret;
}

// cudatoolkit provides __builtin_ctz for NVCC >= 11.0
#if !defined(__CUDACC__) || __CUDACC_VER_MAJOR__ < 11
inline int __builtin_ctz(unsigned long x) {
    unsigned long ret;
    _BitScanForward(&ret, x);
    return (int)ret;
}
#endif

inline int __builtin_clzll(uint64_t x) {
    return (int)__lzcnt64(x);
}

#define __builtin_popcount __popcnt
#define __builtin_popcountl __popcnt64

// MSVC does not define __SSEx__, and _M_IX86_FP is only defined on 32-bit
// processors cf.
// https://docs.microsoft.com/en-us/cpp/preprocessor/predefined-macros
#ifdef __AVX__
#define __SSE__ 1
#define __SSE2__ 1
#define __SSE3__ 1
#define __SSE4_1__ 1
#define __SSE4_2__ 1
#endif

// MSVC sets FMA and F16C automatically when using AVX2
// Ref. FMA (under /arch:AVX2):
// https://docs.microsoft.com/en-us/cpp/build/reference/arch-x64 Ref. F16C (2nd
// paragraph): https://walbourn.github.io/directxmath-avx2/
#ifdef __AVX2__
#define __FMA__ 1
#define __F16C__ 1
#endif

#else
/*******************************************************
 * Linux and OSX
 *******************************************************/

#define FAISS_API
#define posix_memalign_free free

// aligned should be *in front* of the declaration, for compatibility with
// windows
#define ALIGNED(x) __attribute__((aligned(x)))

#endif // _MSC_VER

#if defined(__GNUC__) || defined(__clang__)
#define FAISS_DEPRECATED(msg) __attribute__((deprecated(msg)))
#else
#define FAISS_DEPRECATED(msg)
#endif // GCC or Clang
