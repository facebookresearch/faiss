/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// This file only uses the compiler's default instruction set: CPU-capability
// detection, fp16 (de)coding and packing, and strict kernel dispatch all
// live here so that they are always available even when neither the FMA nor
// the AMX-FP16 kernel below was compiled in.
#include <faiss/impl/scalar_quantizer/sq-fp16-gemm.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/fp16.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__x86_64__)
#include <cpuid.h>
#endif
#if defined(__x86_64__) && defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

// Both are set by CMakeLists.txt only when the corresponding kernel source
// file is part of the build; without them this file must not reference the
// kernel symbols at all, or the library fails to link (e.g. on aarch64).
#ifndef FAISS_FP16_GEMM_FMA_COMPILED
#define FAISS_FP16_GEMM_FMA_COMPILED 0
#endif
#ifndef FAISS_AMX_FP16_COMPILED
#define FAISS_AMX_FP16_COMPILED 0
#endif

namespace faiss {
namespace fp16_gemm {

#if FAISS_FP16_GEMM_FMA_COMPILED
void encode_array_fma(const float*, uint16_t*, size_t);
void gemm_fma_impl(
        const uint16_t*,
        size_t,
        size_t,
        const uint16_t*,
        const uint16_t*,
        size_t,
        size_t,
        size_t,
        size_t,
        float*,
        size_t);
#endif
#if FAISS_AMX_FP16_COMPILED
AMXSubnormalBehavior amx_subnormal_behavior_impl();
void gemm_amx_impl(
        const uint16_t*,
        size_t,
        size_t,
        const uint16_t*,
        const uint16_t*,
        size_t,
        size_t,
        size_t,
        size_t,
        float*,
        size_t);
#endif

namespace {
struct CpuCaps {
    bool fma = false;
    bool amx = false;
};

#if defined(__x86_64__)
uint64_t xgetbv0() {
    uint32_t lo, hi;
    asm volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(lo), "=d"(hi) : "c"(0));
    return (uint64_t{hi} << 32) | lo;
}
#endif

// Checks AVX-512F/BW, F16C and FMA for the FMA kernel, and AMX-TILE plus
// AMX-FP16 for the AMX kernel, including the XCR0 bits that say the OS has
// enabled the corresponding extended state. This is independent from (and
// narrower than) faiss::SIMDConfig: the fp16 gemm kernels are always
// compiled into the main faiss target (see CMakeLists.txt) and pick a
// kernel per-process via CPUID, regardless of which SIMD-level variant of
// the rest of the library was built.
CpuCaps detect_caps() {
    CpuCaps caps;
#if defined(__x86_64__)
    if (__get_cpuid_max(0, nullptr) < 7) {
        return caps;
    }
    unsigned int a, b, c, d;
    __cpuid(1, a, b, c, d);
    if (!(c & (1u << 27))) {
        return caps; // OSXSAVE
    }
    const bool has_fma = (c & (1u << 12)) && (c & (1u << 28)) &&
            (c & (1u << 29)); // FMA, AVX, F16C
    const uint64_t xcr0 = xgetbv0();
    __cpuid_count(7, 0, a, b, c, d);
    const unsigned int max_subleaf = a;
    caps.fma = has_fma && (b & (1u << 16)) && (b & (1u << 30)) &&
            ((xcr0 & 0xe6) == 0xe6); // AVX512F, AVX512BW, ZMM/YMM/XMM state
    const bool tile = (d & (1u << 24)) &&
            ((xcr0 & 0x60000) == 0x60000); // AMX-TILE, tile data/config state
    if (max_subleaf >= 1) {
        __cpuid_count(7, 1, a, b, c, d);
        caps.amx = tile && (a & (1u << 21)); // AMX-FP16
    }
#endif
    return caps;
}

const CpuCaps& cpu_caps() {
    static const CpuCaps caps = detect_caps();
    return caps;
}

// AMX tile usage additionally requires the running thread to have
// requested XTILEDATA permission from the kernel (Linux-specific ABI);
// cached per-thread since a successful request on one thread does not
// extend to others.
bool thread_tile_permission() {
#if defined(__x86_64__) && defined(__linux__)
    static thread_local const bool permitted = [] {
        constexpr unsigned long request_perm = 0x1023; // ARCH_REQ_XCOMP_PERM
        constexpr unsigned long get_perm = 0x1022;     // ARCH_GET_XCOMP_PERM
        constexpr unsigned long tiledata = 18;         // XFEATURE_XTILEDATA
        if (syscall(SYS_arch_prctl, request_perm, tiledata) != 0) {
            return false;
        }
        unsigned long mask = 0;
        return syscall(SYS_arch_prctl, get_perm, &mask) == 0 &&
                (mask & (1ul << tiledata));
    }();
    return permitted;
#else
    return false;
#endif
}
} // namespace

bool amx_fp16_compiled() {
    return FAISS_AMX_FP16_COMPILED != 0;
}
bool fma_available() {
    return FAISS_FP16_GEMM_FMA_COMPILED != 0 && cpu_caps().fma;
}
bool amx_fp16_available() {
    return amx_fp16_compiled() && cpu_caps().amx && thread_tile_permission();
}

bool amx_preserves_subnormals() {
    return amx_subnormal_behavior().preserves;
}

AMXSubnormalBehavior amx_subnormal_behavior() {
    resolve_kernel(GemmKernel::AMX);
#if FAISS_AMX_FP16_COMPILED
    return amx_subnormal_behavior_impl();
#else
    FAISS_THROW_MSG("this build was not compiled with an AMX-FP16 kernel");
#endif
}

GemmKernel resolve_kernel(GemmKernel requested) {
    if (requested == GemmKernel::AUTO) {
        if (amx_fp16_available()) {
            return GemmKernel::AMX;
        }
        if (fma_available()) {
            return GemmKernel::FMA;
        }
        return GemmKernel::SCALAR;
    }
    if (requested == GemmKernel::AMX && !amx_fp16_available()) {
        FAISS_THROW_MSG(
                "explicitly requested AMX-FP16 kernel is not available: "
                "check compiler, CPUID, extended state and thread "
                "permission");
    }
    if (requested == GemmKernel::FMA && !fma_available()) {
        FAISS_THROW_MSG(
                "explicitly requested FP16 FMA kernel is not available: "
                "requires AVX-512F/BW, F16C, FMA and OS support");
    }
    if (requested != GemmKernel::AMX && requested != GemmKernel::FMA &&
        requested != GemmKernel::SCALAR) {
        FAISS_THROW_MSG("unknown fp16 gemm kernel requested");
    }
    return requested;
}

const char* kernel_name(GemmKernel kernel) {
    switch (kernel) {
        case GemmKernel::AUTO:
            return "auto";
        case GemmKernel::AMX:
            return "amx_fp16";
        case GemmKernel::FMA:
            return "fma_fp16";
        case GemmKernel::SCALAR:
            return "scalar_fp16";
    }
    return "unknown";
}

void encode_array(const float* src, uint16_t* dst, size_t n) {
#if FAISS_FP16_GEMM_FMA_COMPILED
    if (fma_available()) {
        encode_array_fma(src, dst, n);
        return;
    }
#endif
    for (size_t i = 0; i < n; ++i) {
        dst[i] = faiss::encode_fp16(src[i]);
    }
}

void encode_rows(const float* src, uint16_t* dst, size_t n, size_t d) {
    encode_array(src, dst, n * d);
}

void decode_array(const uint16_t* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = faiss::decode_fp16(src[i]);
    }
}

void round_values(const float* src, float* dst, size_t n) {
    round_values_range(src, dst, 0, n);
}

void round_values_range(
        const float* src,
        float* dst,
        size_t begin,
        size_t end) {
    FAISS_THROW_IF_NOT_MSG(
            begin % 256 == 0 && begin <= end,
            "fp16 chunked conversion must start on a 256-element boundary "
            "and not go past its end");
    // Converts in small chunks rather than allocating a full fp16 copy of
    // a possibly million-row input.
    uint16_t values[256];
    for (size_t offset = begin; offset < end; offset += 256) {
        const size_t count = end - offset < 256 ? end - offset : 256;
        for (size_t i = 0; i < count; ++i) {
            FAISS_THROW_IF_NOT_FMT(
                    std::isfinite(src[offset + i]),
                    "fp16 input contains NaN or infinity at element %zu",
                    offset + i);
        }
        encode_array(src + offset, values, count);
        for (size_t i = 0; i < count; ++i) {
            FAISS_THROW_IF_NOT_FMT(
                    (values[i] & 0x7c00) != 0x7c00,
                    "fp16 conversion overflowed to infinity at element %zu",
                    offset + i);
            dst[offset + i] = faiss::decode_fp16(values[i]);
        }
    }
}

void round_values_parallel(
        const float* src,
        float* dst,
        size_t n,
        size_t block) {
    FAISS_THROW_IF_NOT_MSG(
            block != 0 && block % 256 == 0,
            "fp16 parallel conversion block size must be a positive "
            "multiple of 256");
    const size_t nblocks = n / block + (n % block != 0);
#ifdef _OPENMP
    if (nblocks >= 2 && !omp_in_parallel() && omp_get_max_threads() > 1) {
        // Each block remembers the first error it hits; at the end we
        // rethrow the one from the lowest-numbered block, since that is
        // the one a serial, start-to-end pass would have hit first.
        // Blocks after an already-failed one don't need to keep going.
        std::vector<std::exception_ptr> errors(nblocks);
        std::atomic<size_t> first_failed{nblocks};
#pragma omp parallel for schedule(static)
        for (size_t b = 0; b < nblocks; ++b) {
            if (b > first_failed.load(std::memory_order_relaxed)) {
                continue;
            }
            const size_t begin = b * block;
            try {
                round_values_range(
                        src, dst, begin, begin + std::min(block, n - begin));
            } catch (...) {
                errors[b] = std::current_exception();
                size_t seen = first_failed.load(std::memory_order_relaxed);
                while (b < seen &&
                       !first_failed.compare_exchange_weak(
                               seen, b, std::memory_order_relaxed)) {
                }
            }
        }
        for (const auto& error : errors) {
            if (error) {
                std::rethrow_exception(error);
            }
        }
        return;
    }
#endif
    round_values(src, dst, n);
}

void validate_codes(const uint8_t* codes, size_t nvalues) {
    for (size_t i = 0; i < nvalues; ++i) {
        uint16_t value;
        std::memcpy(&value, codes + i * 2, 2);
        FAISS_THROW_IF_NOT_MSG(
                (value & 0x7c00) != 0x7c00,
                "fp16 codes must not contain NaN or infinity");
    }
}

void validate_gemm_shape(
        size_t m,
        size_t a_stride,
        size_t b_stride,
        size_t n_padded,
        size_t n_real,
        size_t d,
        size_t out_stride,
        size_t row_multiple) {
    FAISS_THROW_IF_NOT_MSG(
            row_multiple != 0 && m % row_multiple == 0 &&
                    n_padded % kFP16BlockN == 0 && n_real <= n_padded &&
                    a_stride >= d && b_stride >= d && out_stride >= n_padded,
            "invalid fp16 gemm shape or stride");
}

size_t gemm_row_multiple(GemmKernel kernel) {
    return resolve_kernel(kernel) == GemmKernel::AMX ? kFP16TileM : 1;
}

size_t packed_b_size(size_t n_padded, size_t d) {
    FAISS_THROW_IF_NOT_MSG(
            n_padded % kFP16BlockN == 0,
            "fp16 packed centroid count must be a multiple of 32");
    return (d / kFP16TileK) * (n_padded / kFP16TileN) * 512;
}

void pack_centroids(
        const uint16_t* centroids,
        size_t n_real,
        size_t c_stride,
        size_t n_padded,
        size_t d,
        uint16_t* packed) {
    FAISS_THROW_IF_NOT_MSG(
            n_real <= n_padded && c_stride >= d,
            "invalid fp16 centroid packing shape");
    const size_t count = packed_b_size(n_padded, d);
    if (count) {
        std::memset(packed, 0, count * sizeof(uint16_t));
    }
    const size_t kblocks = d / kFP16TileK;
    for (size_t jb = 0; jb < n_padded / kFP16TileN; ++jb) {
        for (size_t kb = 0; kb < kblocks; ++kb) {
            uint16_t* block = packed + (jb * kblocks + kb) * 512;
            for (size_t nn = 0; nn < 16; ++nn) {
                const size_t col = jb * kFP16TileN + nn;
                if (col >= n_real) {
                    continue;
                }
                const uint16_t* src = centroids + col * c_stride + kb * 32;
                for (size_t kk = 0; kk < 16; ++kk) {
                    block[kk * 32 + nn * 2] = src[kk * 2];
                    block[kk * 32 + nn * 2 + 1] = src[kk * 2 + 1];
                }
            }
        }
    }
}

void gemm_scalar(
        const uint16_t* a,
        size_t m,
        size_t a_stride,
        const uint16_t*,
        const uint16_t* b_rows,
        size_t b_stride,
        size_t n_padded,
        size_t n_real,
        size_t d,
        float* out,
        size_t out_stride) {
    validate_gemm_shape(
            m, a_stride, b_stride, n_padded, n_real, d, out_stride, 1);
    for (size_t i = 0; i < m; ++i) {
        float* row = out + i * out_stride;
        for (size_t j = 0; j < n_padded; ++j) {
            row[j] = 0;
        }
        for (size_t j = 0; j < n_real; ++j) {
            float sum = 0;
            for (size_t kk = 0; kk < d; ++kk) {
                sum += faiss::decode_fp16(a[i * a_stride + kk]) *
                        faiss::decode_fp16(b_rows[j * b_stride + kk]);
            }
            row[j] = sum;
        }
    }
}

void gemm_fma(
        const uint16_t* a,
        size_t m,
        size_t a_stride,
        const uint16_t* packed,
        const uint16_t* b_rows,
        size_t b_stride,
        size_t n_padded,
        size_t n_real,
        size_t d,
        float* out,
        size_t out_stride) {
    resolve_kernel(GemmKernel::FMA);
    validate_gemm_shape(
            m, a_stride, b_stride, n_padded, n_real, d, out_stride, 1);
#if FAISS_FP16_GEMM_FMA_COMPILED
    gemm_fma_impl(
            a,
            m,
            a_stride,
            packed,
            b_rows,
            b_stride,
            n_padded,
            n_real,
            d,
            out,
            out_stride);
#else
    // unreachable: resolve_kernel() already throws above when this build has
    // no FMA kernel.
    (void)a;
    (void)packed;
    (void)b_rows;
    (void)out;
    FAISS_THROW_MSG("this build was not compiled with an FP16 FMA kernel");
#endif
}

void gemm_amx(
        const uint16_t* a,
        size_t m,
        size_t a_stride,
        const uint16_t* packed,
        const uint16_t* b_rows,
        size_t b_stride,
        size_t n_padded,
        size_t n_real,
        size_t d,
        float* out,
        size_t out_stride) {
    resolve_kernel(GemmKernel::AMX);
    validate_gemm_shape(
            m, a_stride, b_stride, n_padded, n_real, d, out_stride, kFP16TileM);
#if FAISS_AMX_FP16_COMPILED
    gemm_amx_impl(
            a,
            m,
            a_stride,
            packed,
            b_rows,
            b_stride,
            n_padded,
            n_real,
            d,
            out,
            out_stride);
#else
    // unreachable: resolve_kernel() already throws above when this build has
    // no AMX-FP16 kernel, but keep the pointers "used" so -Wunused-parameter
    // stays clean regardless of FAISS_AMX_FP16_COMPILED.
    (void)a;
    (void)packed;
    (void)b_rows;
    (void)out;
    FAISS_THROW_MSG("this build was not compiled with an AMX-FP16 kernel");
#endif
}

GemmFn get_gemm_fn(GemmKernel kernel) {
    switch (resolve_kernel(kernel)) {
        case GemmKernel::AMX:
            return &gemm_amx;
        case GemmKernel::FMA:
            return &gemm_fma;
        default:
            return &gemm_scalar;
    }
}

} // namespace fp16_gemm
} // namespace faiss
