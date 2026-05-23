/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Implementation of Hamming related functions (distances, smallest distance
 * selection with regular heap|radix and probabilistic heap|radix.
 *
 * IMPLEMENTATION NOTES
 * Optimal speed is typically obtained for vector sizes of multiples of 64
 * bits.
 *
 * hamdis_t is used for distances because at this time
 * it is not clear how we will need to balance
 * - flexibility in vector size (unclear more than 2^16 or even 2^8 bitvectors)
 * - memory usage
 * - cache-misses when dealing with large volumes of data (lower bits is better)
 *
 */

#include <faiss/utils/hamming.h>

#include <cstdio>
#include <cstring>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/simd_dispatch.h>
#include <faiss/utils/utils.h>

// Scalar (NONE) fallback — only needs the generic specializations.
#define THE_SIMD_LEVEL SIMDLevel::NONE
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/hamming_distance/hamming_computer-generic.h>
#include <faiss/utils/hamming_distance/hamming_impl.h>
#undef THE_SIMD_LEVEL

namespace faiss {

size_t hamming_batch_size = 65536;

/******************************************************************
 * Scalar utility functions (no SIMD, no dispatch needed)
 ******************************************************************/

void fvec2bitvec(const float* __restrict x, uint8_t* __restrict b, size_t d) {
    for (size_t i = 0; i < d; i += 8) {
        uint8_t w = 0;
        uint8_t mask = 1;
        size_t nj = i + 8 <= d ? 8 : d - i;
        for (size_t j = 0; j < nj; j++) {
            if (x[i + j] >= 0) {
                w |= mask;
            }
            mask <<= 1;
        }
        *b = w;
        b++;
    }
}

void fvecs2bitvecs(
        const float* __restrict x,
        uint8_t* __restrict b,
        size_t d,
        size_t n) {
    const int64_t ncodes = ((d + 7) / 8);
#pragma omp parallel for if (n > 100000)
    for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
        fvec2bitvec(x + i * d, b + i * ncodes, d);
    }
}

void bitvecs2fvecs(
        const uint8_t* __restrict b,
        float* __restrict x,
        size_t d,
        size_t n) {
    const int64_t ncodes = ((d + 7) / 8);
#pragma omp parallel for if (n > 100000)
    for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
        binary_to_real(d, b + i * ncodes, x + i * d);
    }
}

static uint64_t uint64_reverse_bits(uint64_t b) {
    int i;
    uint64_t revb = 0;
    for (i = 0; i < 64; i++) {
        revb <<= 1;
        revb |= b & 1;
        b >>= 1;
    }
    return revb;
}

void bitvec_print(const uint8_t* b, size_t d) {
    size_t i, j;
    for (i = 0; i < d;) {
        uint64_t brev = uint64_reverse_bits(*(uint64_t*)b);
        for (j = 0; j < 64 && i < d; j++, i++) {
            printf("%d", (int)(brev & 1));
            brev >>= 1;
        }
        b += 8;
        printf(" ");
    }
}

void bitvec_shuffle(
        size_t n,
        size_t da,
        size_t db,
        const int* __restrict order,
        const uint8_t* __restrict a,
        uint8_t* __restrict b) {
    for (size_t i = 0; i < db; i++) {
        FAISS_THROW_IF_NOT(order[i] >= 0 && static_cast<size_t>(order[i]) < da);
    }
    size_t lda = (da + 7) / 8;
    size_t ldb = (db + 7) / 8;

#pragma omp parallel for if (n > 10000)
    for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
        const uint8_t* ai = a + i * lda;
        uint8_t* bi = b + i * ldb;
        memset(bi, 0, ldb);
        for (size_t j = 0; j < db; j++) {
            int o = order[j];
            uint8_t the_bit = (ai[o >> 3] >> (o & 7)) & 1;
            bi[j >> 3] |= the_bit << (j & 7);
        }
    }
}

/******************************************************************
 * Dispatched Hamming distance public API
 ******************************************************************/

void hammings(
        const uint8_t* __restrict a,
        const uint8_t* __restrict b,
        size_t na,
        size_t nb,
        size_t ncodes,
        hamdis_t* __restrict dis) {
    with_simd_level([&]<SIMDLevel SL>() {
        hammings_fixSL<SL>(a, b, na, nb, ncodes, dis);
    });
}

void hammings_knn(
        int_maxheap_array_t* __restrict ha,
        const uint8_t* __restrict a,
        const uint8_t* __restrict b,
        size_t nb,
        size_t ncodes,
        int order) {
    hammings_knn_hc(ha, a, b, nb, ncodes, order);
}

void hammings_knn_hc(
        int_maxheap_array_t* __restrict ha,
        const uint8_t* __restrict a,
        const uint8_t* __restrict b,
        size_t nb,
        size_t ncodes,
        int order,
        ApproxTopK_mode_t approx_topk_mode,
        const faiss::IDSelector* sel) {
    with_simd_level([&]<SIMDLevel SL>() {
        hammings_knn_hc_fixSL<SL>(
                ha, a, b, nb, ncodes, order, approx_topk_mode, sel);
    });
}

void hammings_knn_mc(
        const uint8_t* __restrict a,
        const uint8_t* __restrict b,
        size_t na,
        size_t nb,
        size_t k,
        size_t ncodes,
        int32_t* __restrict distances,
        int64_t* __restrict labels,
        const faiss::IDSelector* sel) {
    with_simd_level([&]<SIMDLevel SL>() {
        hammings_knn_mc_fixSL<SL>(
                a, b, na, nb, k, ncodes, distances, labels, sel);
    });
}

void hamming_range_search(
        const uint8_t* a,
        const uint8_t* b,
        size_t na,
        size_t nb,
        int radius,
        size_t code_size,
        RangeSearchResult* result,
        const faiss::IDSelector* sel) {
    with_simd_level([&]<SIMDLevel SL>() {
        hamming_range_search_fixSL<SL>(
                a, b, na, nb, radius, code_size, result, sel);
    });
}

void hamming_count_thres(
        const uint8_t* bs1,
        const uint8_t* bs2,
        size_t n1,
        size_t n2,
        hamdis_t ht,
        size_t ncodes,
        size_t* nptr) {
    with_simd_level([&]<SIMDLevel SL>() {
        hamming_count_thres_fixSL<SL>(bs1, bs2, n1, n2, ht, ncodes, nptr);
    });
}

void crosshamming_count_thres(
        const uint8_t* dbs,
        size_t n,
        hamdis_t ht,
        size_t ncodes,
        size_t* nptr) {
    with_simd_level([&]<SIMDLevel SL>() {
        crosshamming_count_thres_fixSL<SL>(dbs, n, ht, ncodes, nptr);
    });
}

size_t match_hamming_thres(
        const uint8_t* bs1,
        const uint8_t* bs2,
        size_t n1,
        size_t n2,
        hamdis_t ht,
        size_t ncodes,
        int64_t* idx,
        hamdis_t* dis) {
    return with_simd_level([&]<SIMDLevel SL>() -> size_t {
        return match_hamming_thres_fixSL<SL>(
                bs1, bs2, n1, n2, ht, ncodes, idx, dis);
    });
}

void generalized_hammings_knn_hc(
        int_maxheap_array_t* __restrict ha,
        const uint8_t* __restrict a,
        const uint8_t* __restrict b,
        size_t nb,
        size_t code_size,
        int ordered) {
    with_simd_level([&]<SIMDLevel SL>() {
        generalized_hammings_knn_hc_fixSL<SL>(ha, a, b, nb, code_size, ordered);
    });
}

/******************************************************************
 * Bitstring pack/unpack (scalar, no dispatch needed)
 ******************************************************************/

void pack_bitstrings(
        size_t n,
        size_t M,
        int nbit,
        const int32_t* unpacked,
        uint8_t* packed,
        size_t code_size) {
    FAISS_THROW_IF_NOT(code_size >= (M * nbit + 7) / 8);
#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
        const int32_t* in = unpacked + i * M;
        uint8_t* out = packed + i * code_size;
        BitstringWriter wr(out, code_size);
        for (size_t j = 0; j < M; j++) {
            wr.write(in[j], nbit);
        }
    }
}

void pack_bitstrings(
        size_t n,
        size_t M,
        const int32_t* nbit,
        const int32_t* unpacked,
        uint8_t* packed,
        size_t code_size) {
    int totbit = 0;
    for (size_t j = 0; j < M; j++) {
        totbit += nbit[j];
    }
    FAISS_THROW_IF_NOT(code_size >= static_cast<size_t>((totbit + 7) / 8));
#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
        const int32_t* in = unpacked + i * M;
        uint8_t* out = packed + i * code_size;
        BitstringWriter wr(out, code_size);
        for (size_t j = 0; j < M; j++) {
            wr.write(in[j], nbit[j]);
        }
    }
}

void unpack_bitstrings(
        size_t n,
        size_t M,
        int nbit,
        const uint8_t* packed,
        size_t code_size,
        int32_t* unpacked) {
    FAISS_THROW_IF_NOT(code_size >= (M * nbit + 7) / 8);
#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
        const uint8_t* in = packed + i * code_size;
        int32_t* out = unpacked + i * M;
        BitstringReader rd(in, code_size);
        for (size_t j = 0; j < M; j++) {
            out[j] = rd.read(nbit);
        }
    }
}

void unpack_bitstrings(
        size_t n,
        size_t M,
        const int32_t* nbit,
        const uint8_t* packed,
        size_t code_size,
        int32_t* unpacked) {
    int totbit = 0;
    for (size_t j = 0; j < M; j++) {
        totbit += nbit[j];
    }
    FAISS_THROW_IF_NOT(code_size >= static_cast<size_t>((totbit + 7) / 8));
#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
        const uint8_t* in = packed + i * code_size;
        int32_t* out = unpacked + i * M;
        BitstringReader rd(in, code_size);
        for (size_t j = 0; j < M; j++) {
            out[j] = rd.read(nbit[j]);
        }
    }
}

} // namespace faiss
