/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/rabitq_integer_adc.h>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#if defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(_WIN32)
#include <windows.h>
#endif
#endif

namespace faiss::rabitq_integer_adc {

namespace {

bool detect_avx512_vnni() {
#if defined(__x86_64__) || defined(_M_X64)
    [[maybe_unused]] unsigned int a, b, c, d;
#if defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, 0, 0);
    if (regs[0] < 7) {
        return false;
    }
    __cpuidex(regs, 1, 0);
    c = static_cast<unsigned int>(regs[2]);
#else
    if (!__get_cpuid(1, &a, &b, &c, &d)) {
        return false;
    }
#endif
    // OSXSAVE and AVX are required before reading the enabled XCR0 state.
    if ((c & ((1u << 27) | (1u << 28))) != ((1u << 27) | (1u << 28))) {
        return false;
    }
#if defined(_MSC_VER)
    const uint64_t xcr0 = _xgetbv(0);
    __cpuidex(regs, 7, 0);
    b = static_cast<unsigned int>(regs[1]);
    c = static_cast<unsigned int>(regs[2]);
#else
    unsigned int xcr0_low, xcr0_high;
    asm volatile("xgetbv" : "=a"(xcr0_low), "=d"(xcr0_high) : "c"(0));
    const uint64_t xcr0 = uint64_t(xcr0_low) | (uint64_t(xcr0_high) << 32);
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) {
        return false;
    }
#endif
    constexpr uint64_t state_mask = 0xe6; // XMM, YMM and AVX-512 state.
    // AVX512F, DQ, CD, BW and VL match the isolated kernel's compiler flags.
    constexpr unsigned int base_mask =
            (1u << 16) | (1u << 17) | (1u << 28) | (1u << 30) | (1u << 31);
    constexpr unsigned int integer_mask = (1u << 1) | (1u << 11); // VBMI, VNNI.
    return (xcr0 & state_mask) == state_mask && (b & base_mask) == base_mask &&
            (c & integer_mask) == integer_mask;
#else
    return false;
#endif
}

} // namespace

bool avx512_vnni_supported() {
    static const bool supported = detect_avx512_vnni();
    return supported;
}

bool arm_dotprod_supported() {
#if defined(__aarch64__) || defined(_M_ARM64)
#if defined(__linux__) && defined(HWCAP_ASIMDDP)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#elif defined(__APPLE__)
    int supported = 0;
    size_t size = sizeof(supported);
    return sysctlbyname(
                   "hw.optional.arm.FEAT_DotProd",
                   &supported,
                   &size,
                   nullptr,
                   0) == 0 &&
            supported != 0;
#elif defined(_WIN32) && defined(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE)
    return IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE) != 0;
#else
    return false;
#endif
#else
    return false;
#endif
}

int64_t dot_product_scalar(
        const int8_t* query,
        const int8_t* levels,
        size_t d) {
    int64_t result = 0;
    for (size_t j = 0; j < d; j++) {
        result += int64_t(query[j]) * int64_t(levels[j]);
    }
    return result;
}

void dot_product_batch_4_scalar(
        const int8_t* query,
        const int8_t* levels0,
        const int8_t* levels1,
        const int8_t* levels2,
        const int8_t* levels3,
        size_t d,
        int64_t& dot0,
        int64_t& dot1,
        int64_t& dot2,
        int64_t& dot3) {
    dot0 = dot1 = dot2 = dot3 = 0;
    for (size_t j = 0; j < d; j++) {
        const int64_t q = query[j];
        dot0 += q * levels0[j];
        dot1 += q * levels1[j];
        dot2 += q * levels2[j];
        dot3 += q * levels3[j];
    }
}

void dot_product_batch_8_scalar(
        const int8_t* query,
        const int8_t* const levels[8],
        size_t d,
        int64_t dots[8]) {
    for (size_t k = 0; k < 8; ++k) {
        dots[k] = 0;
    }
    for (size_t j = 0; j < d; ++j) {
        const int64_t q = query[j];
        for (size_t k = 0; k < 8; ++k) {
            dots[k] += q * levels[k][j];
        }
    }
}

void dot_product_batch_16_scalar(
        const int8_t* query,
        const int8_t* const levels[16],
        size_t d,
        int64_t dots[16]) {
    for (size_t k = 0; k < 16; ++k) {
        dots[k] = 0;
    }
    for (size_t j = 0; j < d; ++j) {
        const int64_t q = query[j];
        for (size_t k = 0; k < 16; ++k) {
            dots[k] += q * levels[k][j];
        }
    }
}

void dot_product_batch_tail_scalar(
        const int8_t* query,
        const int8_t* const* levels,
        int count,
        size_t d,
        int64_t* dots) {
    for (int k = 0; k < count; ++k) {
        dots[k] = 0;
    }
    for (size_t j = 0; j < d; ++j) {
        const int64_t q = query[j];
        for (int k = 0; k < count; ++k) {
            dots[k] += q * levels[k][j];
        }
    }
}

} // namespace faiss::rabitq_integer_adc
