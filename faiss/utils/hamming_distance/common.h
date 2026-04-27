/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef FAISS_hamming_common_h
#define FAISS_hamming_common_h

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <faiss/impl/platform_macros.h>
#include <faiss/utils/popcount.h>
#include <faiss/utils/simd_levels.h>

#ifdef __aarch64__
// Include <arm_neon.h> at global scope so the NEON types it declares
// (uint8x16_t, uint16x8_t, ...) end up in `::` and not inside `faiss::`.
#include <arm_neon.h>
#endif

/* The Hamming distance type */
using hamdis_t = int32_t;

namespace faiss {

// This table was moved from .cpp to .h file, because
// otherwise it was causing compilation errors while trying to
// compile swig modules on Windows.
inline constexpr uint8_t hamdis_tab_ham_bytes[256] = {
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4,
        2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4,
        2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6,
        4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5,
        3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6,
        4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};

/***************************************************************************
 * generalized Hamming = number of bytes that are different between
 * two codes.
 ***************************************************************************/

inline int generalized_hamming_64(uint64_t a) {
    a |= a >> 1;
    a |= a >> 2;
    a |= a >> 4;
    a &= 0x0101010101010101UL;
    return popcount64(a);
}

/***************************************************************************
 * Free-function hamming<nbits>() templates.
 * hamming<64> is architecture-independent.
 * hamming<128>, hamming<256>, the primary template, and the runtime-nwords
 * overload have NEON-optimized versions on aarch64.
 ***************************************************************************/

#ifndef SWIG

/* hamming<64> — identical on all architectures */
template <size_t nbits>
inline hamdis_t hamming(const uint64_t* bs1, const uint64_t* bs2);

template <>
inline hamdis_t hamming<64>(const uint64_t* pa, const uint64_t* pb) {
    return popcount64(pa[0] ^ pb[0]);
}

#ifdef __aarch64__

/* Hamming distances for multiples of 64 bits — NEON version */
template <size_t nbits>
inline hamdis_t hamming(const uint64_t* pa, const uint64_t* pb) {
    constexpr size_t nwords256 = nbits / 256;
    constexpr size_t nwords128 = (nbits - nwords256 * 256) / 128;
    constexpr size_t nwords64 =
            (nbits - nwords256 * 256 - nwords128 * 128) / 64;

    hamdis_t h = 0;
    if (nwords256 > 0) {
        for (size_t i = 0; i < nwords256; i++) {
            h += hamming<256>(pa, pb);
            pa += 4;
            pb += 4;
        }
    }

    if (nwords128 > 0) {
        h += hamming<128>(pa, pb);
        pa += 2;
        pb += 2;
    }

    if (nwords64 > 0) {
        h += hamming<64>(pa, pb);
    }

    return h;
}

template <>
inline hamdis_t hamming<128>(const uint64_t* pa, const uint64_t* pb) {
    const uint8_t* pa8 = reinterpret_cast<const uint8_t*>(pa);
    const uint8_t* pb8 = reinterpret_cast<const uint8_t*>(pb);
    uint8x16_t or0 = veorq_u8(vld1q_u8(pa8), vld1q_u8(pb8));
    uint8x16_t c0 = vcntq_u8(or0);
    auto dis = vaddvq_u8(c0);
    return dis;
}

template <>
inline hamdis_t hamming<256>(const uint64_t* pa, const uint64_t* pb) {
    const uint8_t* pa8 = reinterpret_cast<const uint8_t*>(pa);
    const uint8_t* pb8 = reinterpret_cast<const uint8_t*>(pb);
    uint8x16_t or0 = veorq_u8(vld1q_u8(pa8), vld1q_u8(pb8));
    uint8x16_t or1 = veorq_u8(vld1q_u8(pa8 + 16), vld1q_u8(pb8 + 16));
    uint8x16_t c0 = vcntq_u8(or0);
    uint8x16_t c1 = vcntq_u8(or1);
    uint8x16_t ca = vpaddq_u8(c0, c1);
    auto dis = vaddvq_u8(ca);
    return dis;
}

/* Hamming distances for multiple of 64 bits */
inline hamdis_t hamming(const uint64_t* pa, const uint64_t* pb, size_t nwords) {
    const size_t nwords256 = nwords / 4;
    const size_t nwords128 = (nwords % 4) / 2;
    const size_t nwords64 = nwords % 2;

    hamdis_t h = 0;
    if (nwords256 > 0) {
        for (size_t i = 0; i < nwords256; i++) {
            h += hamming<256>(pa, pb);
            pa += 4;
            pb += 4;
        }
    }

    if (nwords128 > 0) {
        h += hamming<128>(pa, pb);
        pa += 2;
        pb += 2;
    }

    if (nwords64 > 0) {
        h += hamming<64>(pa, pb);
    }

    return h;
}

#else // !__aarch64__

/* Hamming distances for multiples of 64 bits — scalar version */
template <size_t nbits>
inline hamdis_t hamming(const uint64_t* bs1, const uint64_t* bs2) {
    const size_t nwords = nbits / 64;
    size_t i;
    hamdis_t h = 0;
    for (i = 0; i < nwords; i++) {
        h += popcount64(bs1[i] ^ bs2[i]);
    }
    return h;
}

/* specialized (optimized) functions */
template <>
inline hamdis_t hamming<128>(const uint64_t* pa, const uint64_t* pb) {
    return popcount64(pa[0] ^ pb[0]) + popcount64(pa[1] ^ pb[1]);
}

template <>
inline hamdis_t hamming<256>(const uint64_t* pa, const uint64_t* pb) {
    return popcount64(pa[0] ^ pb[0]) + popcount64(pa[1] ^ pb[1]) +
            popcount64(pa[2] ^ pb[2]) + popcount64(pa[3] ^ pb[3]);
}

/* Hamming distances for multiple of 64 bits */
inline hamdis_t hamming(
        const uint64_t* bs1,
        const uint64_t* bs2,
        size_t nwords) {
    hamdis_t h = 0;
    for (size_t i = 0; i < nwords; i++) {
        h += popcount64(bs1[i] ^ bs2[i]);
    }
    return h;
}

#endif // __aarch64__

/***************************************************************************
 * Bit-level Hamming distance implementation functions.
 * These depend only on the hamming<nbits>() free functions above.
 ***************************************************************************/

template <size_t nbits>
inline void hammings_impl(
        const uint64_t* __restrict bs1,
        const uint64_t* __restrict bs2,
        size_t n1,
        size_t n2,
        hamdis_t* __restrict dis) {
    size_t i, j;
    const size_t nwords = nbits / 64;
    for (i = 0; i < n1; i++) {
        const uint64_t* __restrict bs1_ = bs1 + i * nwords;
        hamdis_t* __restrict dis_ = dis + i * n2;
        for (j = 0; j < n2; j++) {
            dis_[j] = hamming<nbits>(bs1_, bs2 + j * nwords);
        }
    }
}

inline void hammings_impl_runtime(
        const uint64_t* __restrict bs1,
        const uint64_t* __restrict bs2,
        size_t n1,
        size_t n2,
        size_t nbits,
        hamdis_t* __restrict dis) {
    size_t i, j;
    const size_t nwords = nbits / 64;
    for (i = 0; i < n1; i++) {
        const uint64_t* __restrict bs1_ = bs1 + i * nwords;
        hamdis_t* __restrict dis_ = dis + i * n2;
        for (j = 0; j < n2; j++) {
            dis_[j] = hamming(bs1_, bs2 + j * nwords, nwords);
        }
    }
}

template <size_t nbits>
inline void hamming_count_thres_impl(
        const uint64_t* __restrict bs1,
        const uint64_t* __restrict bs2,
        size_t n1,
        size_t n2,
        hamdis_t ht,
        size_t* __restrict nptr) {
    const size_t nwords = nbits / 64;
    size_t i, j, posm = 0;
    const uint64_t* bs2_ = bs2;

    for (i = 0; i < n1; i++) {
        bs2 = bs2_;
        for (j = 0; j < n2; j++) {
            if (hamming<nbits>(bs1, bs2) <= ht) {
                posm++;
            }
            bs2 += nwords;
        }
        bs1 += nwords;
    }
    *nptr = posm;
}

template <size_t nbits>
inline void crosshamming_count_thres_impl(
        const uint64_t* __restrict dbs,
        size_t n,
        int ht,
        size_t* __restrict nptr) {
    const size_t nwords = nbits / 64;
    size_t i, j, posm = 0;
    const uint64_t* bs1 = dbs;
    for (i = 0; i < n; i++) {
        const uint64_t* bs2 = bs1 + 2;
        for (j = i + 1; j < n; j++) {
            if (hamming<nbits>(bs1, bs2) <= ht) {
                posm++;
            }
            bs2 += nwords;
        }
        bs1 += nwords;
    }
    *nptr = posm;
}

template <size_t nbits>
inline size_t match_hamming_thres_impl(
        const uint64_t* __restrict bs1,
        const uint64_t* __restrict bs2,
        size_t n1,
        size_t n2,
        int ht,
        int64_t* __restrict idx,
        hamdis_t* __restrict hams) {
    const size_t nwords = nbits / 64;
    size_t i, j, posm = 0;
    hamdis_t h;
    const uint64_t* bs2_ = bs2;
    for (i = 0; i < n1; i++) {
        bs2 = bs2_;
        for (j = 0; j < n2; j++) {
            h = hamming<nbits>(bs1, bs2);
            if (h <= ht) {
                *idx = i;
                idx++;
                *idx = j;
                idx++;
                *hams = h;
                hams++;
                posm++;
            }
            bs2 += nwords;
        }
        bs1 += nwords;
    }
    return posm;
}

#endif // SWIG

} // namespace faiss

#endif
