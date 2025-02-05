/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include <limits.h>

#ifndef __clang__
inline int __builtin_ctzll(uint64_t x) {
    unsigned long ret;
    _BitScanForward64(&ret, x);
    return (int)ret;
}
#endif

// cudatoolkit provides __builtin_ctz for NVCC >= 11.0
#if !defined(__CUDACC__) || __CUDACC_VER_MAJOR__ < 11
inline int __builtin_ctz(unsigned long x) {
    unsigned long ret;
    _BitScanForward(&ret, x);
    return (int)ret;
}
#endif

#ifndef __clang__
inline int __builtin_clzll(uint64_t x) {
#if defined(_M_X64) || defined(__x86_64__)
    return (int)__lzcnt64(x);
#elif defined(_M_ARM64)
    unsigned long index;
    int count = sizeof(uint64_t) * CHAR_BIT;
    if (_BitScanReverse64(&index, x)) {
        count = count - 1 - index;
    }
    return count;
#endif
}
#endif

#define __builtin_popcount __popcnt
#define __builtin_popcountl __popcnt64

#ifndef __clang__
#define __m128i_u __m128i
#define __m256i_u __m256i
#endif

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

#define FAISS_ALWAYS_INLINE __forceinline

#else
/*******************************************************
 * Linux and OSX
 *******************************************************/

#define FAISS_API
#define posix_memalign_free free

// aligned should be *in front* of the declaration, for compatibility with
// windows
#ifdef SWIG
#define ALIGNED(x)
#else
#define ALIGNED(x) __attribute__((aligned(x)))
#endif

#define FAISS_ALWAYS_INLINE __attribute__((always_inline)) inline

#endif

#if defined(__GNUC__) || defined(__clang__)
#define FAISS_DEPRECATED(msg) __attribute__((deprecated(msg)))
#else
#define FAISS_DEPRECATED(msg)
#endif // GCC or Clang

// Localized enablement of imprecise floating point operations
// You need to use all 3 macros to cover all compilers.
#if defined(_MSC_VER)
#define FAISS_PRAGMA_IMPRECISE_LOOP
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN \
    __pragma(float_control(precise, off, push))
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_END __pragma(float_control(pop))
#elif defined(__clang__)
#if defined(__PPC__)
#define FAISS_PRAGMA_IMPRECISE_LOOP \
    _Pragma("clang loop vectorize_width(4) interleave_count(8)")
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN \
    _Pragma("float_control(precise, off, push)")
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_END _Pragma("float_control(pop)")
#else
#define FAISS_PRAGMA_IMPRECISE_LOOP \
    _Pragma("clang loop vectorize(enable) interleave(enable)")

// clang-format off

// the following ifdef is needed, because old versions of clang (prior to 14)
// do not generate FMAs on x86 unless this pragma is used. On the other hand,
// ARM does not support the following pragma flag.
// TODO: find out how to enable FMAs on clang 10 and earlier.
#if defined(__x86_64__) && (defined(__clang_major__) && (__clang_major__ > 10))
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN \
    _Pragma("float_control(precise, off, push)")
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_END _Pragma("float_control(pop)")
#else
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_END
#endif
#endif
#elif defined(__GNUC__)
// Unfortunately, GCC does not provide a pragma for detecting it.
// So, we have to stick to GNUC, which is defined by MANY compilers.
// This is why clang/icc needs to be checked first.

// todo: add __INTEL_COMPILER check for the classic ICC
// todo: add __INTEL_LLVM_COMPILER for ICX

#define FAISS_PRAGMA_IMPRECISE_LOOP
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN \
    _Pragma("GCC push_options") \
    _Pragma("GCC optimize (\"unroll-loops,associative-math,no-signed-zeros\")")
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_END \
    _Pragma("GCC pop_options")
#else
#define FAISS_PRAGMA_IMPRECISE_LOOP
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_END
#endif

// clang-format on

/*******************************************************
 * BIGENDIAN specific macros
 *******************************************************/
#if !defined(_MSC_VER) && \
        (defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__))
#define FAISS_BIG_ENDIAN
#endif

#define Swap2Bytes(val) ((((val) >> 8) & 0x00FF) | (((val) << 8) & 0xFF00))

#define Swap4Bytes(val)                                           \
    ((((val) >> 24) & 0x000000FF) | (((val) >> 8) & 0x0000FF00) | \
     (((val) << 8) & 0x00FF0000) | (((val) << 24) & 0xFF000000))

/*******************************************************
 * Fake structure to detect structure packing/alignment 
 * issues.
 *******************************************************/

#include <vector> 

namespace faiss {

struct StructPackingTestD {
    int a; 
    bool b, c, d; 
    long e; 
    bool f, g, h, i, j; 
};

struct StructPackingTestC : StructPackingTestD {
    bool k; 
    int l; 
    bool m;
};

struct StructPackingTestA {
    virtual void* operator()(int) {return nullptr;}

    virtual ~StructPackingTestA() {}
};

struct StructPackingTestB : StructPackingTestA {
    StructPackingTestC options;
    std::vector<void*> vres;
    std::vector<int> devices;
    int ncall;

    void* operator()(int) override { return nullptr;}
    virtual ~StructPackingTestB() {}

};

/// How user vector index data is stored on the GPU
enum IndicesOptionsB {
    /// The user indices are only stored on the CPU; the GPU returns
    /// (inverted list, offset) to the CPU which is then translated to
    /// the real user index.
    INDICES_CPU = 0,
    /// The indices are not stored at all, on either the CPU or
    /// GPU. Only (inverted list, offset) is returned to the user as the
    /// index.
    INDICES_IVF = 1,
    /// Indices are stored as 32 bit integers on the GPU, but returned
    /// as 64 bit integers
    INDICES_32_BIT = 2,
    /// Indices are stored as 64 bit integers on the GPU
    INDICES_64_BIT = 3,
};

/// set some options on how to copy to GPU
struct GpuClonerOptionsB {
    /// how should indices be stored on index types that support indices
    /// (anything but GpuIndexFlat*)?
    IndicesOptionsB indicesOptions = INDICES_64_BIT;

    /// is the coarse quantizer in float16?
    bool useFloat16CoarseQuantizer = false;

    /// for GpuIndexIVFFlat, is storage in float16?
    /// for GpuIndexIVFPQ, are intermediate calculations in float16?
    bool useFloat16 = false;

    /// use precomputed tables?
    bool usePrecomputed = false;

    /// reserve vectors in the invfiles?
    long reserveVecs = 0;

    /// For GpuIndexFlat, store data in transposed layout?
    bool storeTransposed = false;

    /// Set verbose options on the index
    bool verbose = false;

    /// use the cuVS implementation
#if defined USE_NVIDIA_CUVS
    bool use_cuvs = true;
#else
    bool use_cuvs = false;
#endif

    /// This flag controls the CPU fallback logic for coarse quantizer
    /// component of the index. When set to false (default), the cloner will
    /// throw an exception for indices not implemented on GPU. When set to
    /// true, it will fallback to a CPU implementation.
    bool allowCpuCoarseQuantizer = false;
};

struct GpuMultipleClonerOptionsB : public GpuClonerOptionsB {
    /// Whether to shard the index across GPUs, versus replication
    /// across GPUs
    bool shard = false;

    /// IndexIVF::copy_subset_to subset type
    int shard_type = 1;

    /// set to true if an IndexIVF is to be dispatched to multiple GPUs with a
    /// single common IVF quantizer, ie. only the inverted lists are sharded on
    /// the sub-indexes (uses an IndexShardsIVF)
    bool common_ivf_quantizer = false;
};

struct ProgressiveDimIndexFactoryB {
    /// ownership transferred to caller
    virtual void* operator()(int dim);

    virtual ~ProgressiveDimIndexFactoryB() {}
};

struct GpuProgressiveDimIndexFactoryB : ProgressiveDimIndexFactoryB {
    GpuMultipleClonerOptionsB options;
    std::vector<void*> vres;
    std::vector<int> devices;
    int ncall;

    explicit GpuProgressiveDimIndexFactoryB(int ngpu);

    void* operator()(int dim) override {return nullptr; }

    virtual ~GpuProgressiveDimIndexFactoryB() override {}

    void say();
};


// body of function should be 
// int function_name (int q) STRUCT_PACKING_FUNCTION_BODY
#define STRUCT_PACKING_FUNCTION_BODY {  \
    StructPackingTestB sb; \
    switch(q) {  \
    case 0:   \
        return sizeof(StructPackingTestB);  \
    case 1:  \
        return (char*)&sb.ncall - (char*)&sb;  \
    case 2:  \
        return sizeof(gpu::GpuProgressiveDimIndexFactoryB);  \
    default: \
        return -1; \
    } \
}


} 