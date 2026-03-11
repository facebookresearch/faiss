/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/fast_scan/LookupTableScaler.h>
#include <faiss/impl/fast_scan/decompose_qbs.h>
#include <faiss/impl/fast_scan/fast_scan.h>
#include <faiss/impl/fast_scan/simd_result_handlers.h>
#include <faiss/impl/simd_dispatch.h>

#include <array>

namespace faiss {

using namespace simd_result_handlers;

/***************************************************************
 * Packing functions for codes
 ***************************************************************/

namespace {

/* extract the column starting at (i, j)
 * from packed matrix src of size (m, n)*/
template <typename T, class TA>
void get_matrix_column(
        T* src,
        size_t m,
        size_t n,
        int64_t i,
        int64_t j,
        TA& dest) {
    for (int64_t k = 0; k < dest.size(); k++) {
        if (k + i >= 0 && k + i < m) {
            dest[k] = src[(k + i) * n + j];
        } else {
            dest[k] = 0;
        }
    }
}

} // anonymous namespace

void pq4_pack_codes(
        const uint8_t* codes,
        size_t ntotal,
        size_t M,
        size_t nb,
        size_t bbs,
        size_t nsq,
        uint8_t* blocks,
        size_t code_stride) {
    // Determine stride: use custom if provided, otherwise use legacy
    // calculation
    size_t actual_stride = (code_stride == 0) ? (M + 1) / 2 : code_stride;

    // Input validation for custom stride
    if (code_stride != 0) {
        FAISS_THROW_IF_NOT_MSG(
                code_stride >= (M + 1) / 2,
                "Custom stride must be >= minimum code size");
    }

    FAISS_THROW_IF_NOT(bbs % 32 == 0);
    FAISS_THROW_IF_NOT(nb % bbs == 0);
    FAISS_THROW_IF_NOT(nsq % 2 == 0);

    if (nb == 0) {
        return;
    }
    memset(blocks, 0, nb * nsq / 2);
#ifdef FAISS_BIG_ENDIAN
    const uint8_t perm0[16] = {
            8, 0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7};
#else
    const uint8_t perm0[16] = {
            0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
#endif

    uint8_t* codes2 = blocks;
    for (size_t i0 = 0; i0 < nb; i0 += bbs) {
        for (int sq = 0; sq < nsq; sq += 2) {
            for (size_t i = 0; i < bbs; i += 32) {
                std::array<uint8_t, 32> c, c0, c1;
                get_matrix_column(
                        codes, ntotal, actual_stride, i0 + i, sq / 2, c);

                for (int j = 0; j < 32; j++) {
                    c0[j] = c[j] & 15;
                    c1[j] = c[j] >> 4;
                }
                for (int j = 0; j < 16; j++) {
                    uint8_t d0, d1;
                    d0 = c0[perm0[j]] | (c0[perm0[j] + 16] << 4);
                    d1 = c1[perm0[j]] | (c1[perm0[j] + 16] << 4);
                    codes2[j] = d0;
                    codes2[j + 16] = d1;
                }
                codes2 += 32;
            }
        }
    }
}

void pq4_pack_codes_range(
        const uint8_t* codes,
        size_t M,
        size_t i0,
        size_t i1,
        size_t bbs,
        size_t nsq,
        uint8_t* blocks,
        size_t code_stride,
        size_t block_stride) {
    // Determine stride: use custom if provided, otherwise use legacy
    // calculation
    size_t actual_stride = (code_stride == 0) ? (M + 1) / 2 : code_stride;

    // Input validation for custom stride
    if (code_stride != 0) {
        FAISS_THROW_IF_NOT_MSG(
                code_stride >= (M + 1) / 2,
                "Custom stride must be >= minimum code size");
    }

#ifdef FAISS_BIG_ENDIAN
    const uint8_t perm0[16] = {
            8, 0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7};
#else
    const uint8_t perm0[16] = {
            0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
#endif

    // range of affected blocks
    size_t block0 = i0 / bbs;
    size_t block1 = ((i1 - 1) / bbs) + 1;

    for (size_t b = block0; b < block1; b++) {
        uint8_t* codes2 = blocks + b * block_stride;
        int64_t i_base = b * bbs - i0;
        for (int sq = 0; sq < nsq; sq += 2) {
            for (size_t i = 0; i < bbs; i += 32) {
                std::array<uint8_t, 32> c, c0, c1;
                get_matrix_column(
                        codes, i1 - i0, actual_stride, i_base + i, sq / 2, c);

                for (int j = 0; j < 32; j++) {
                    c0[j] = c[j] & 15;
                    c1[j] = c[j] >> 4;
                }
                for (int j = 0; j < 16; j++) {
                    uint8_t d0, d1;
                    d0 = c0[perm0[j]] | (c0[perm0[j] + 16] << 4);
                    d1 = c1[perm0[j]] | (c1[perm0[j] + 16] << 4);
                    codes2[j] |= d0;
                    codes2[j + 16] |= d1;
                }
                codes2 += 32;
            }
        }
    }
}

namespace {

// get the specific address of the vector inside a block
// shift is used for determine the if the saved in bits 0..3 (false) or
// bits 4..7 (true)
size_t get_vector_specific_address(
        size_t bbs,
        size_t vector_id,
        size_t sq,
        bool& shift) {
    // get the vector_id inside the block
    vector_id = vector_id % bbs;
    shift = vector_id > 15;
    vector_id = vector_id & 15;

    // get the address of the vector in sq
    size_t address;
    if (vector_id < 8) {
        address = vector_id << 1;
    } else {
        address = ((vector_id - 8) << 1) + 1;
    }
    if (sq & 1) {
        address += 16;
    }
    return (sq >> 1) * bbs + address;
}

} // anonymous namespace

uint8_t pq4_get_packed_element(
        const uint8_t* data,
        size_t bbs,
        size_t nsq,
        size_t vector_id,
        size_t sq) {
    // move to correct bbs-sized block
    // number of blocks * block size
    data += (vector_id / bbs) * (((nsq + 1) / 2) * bbs);
    bool shift;
    size_t address = get_vector_specific_address(bbs, vector_id, sq, shift);
    if (shift) {
        return data[address] >> 4;
    } else {
        return data[address] & 15;
    }
}

void pq4_set_packed_element(
        uint8_t* data,
        uint8_t code,
        size_t bbs,
        size_t nsq,
        size_t vector_id,
        size_t sq) {
    // move to correct bbs-sized block
    // number of blocks * block size
    data += (vector_id / bbs) * (((nsq + 1) / 2) * bbs);
    bool shift;
    size_t address = get_vector_specific_address(bbs, vector_id, sq, shift);
    if (shift) {
        data[address] = (code << 4) | (data[address] & 15);
    } else {
        data[address] = code | (data[address] & ~15);
    }
}

/***************************************************************
 * CodePackerPQ4 implementation
 ***************************************************************/

CodePackerPQ4::CodePackerPQ4(size_t nsq, size_t bbs) {
    this->nsq = nsq;
    nvec = bbs;
    code_size = (nsq * 4 + 7) / 8;
    block_size = ((nsq + 1) / 2) * bbs;
}

void CodePackerPQ4::pack_1(
        const uint8_t* flat_code,
        size_t offset,
        uint8_t* block) const {
    size_t bbs = nvec;
    if (offset >= nvec) {
        block += (offset / nvec) * block_size;
        offset = offset % nvec;
    }
    for (size_t i = 0; i < code_size; i++) {
        uint8_t code = flat_code[i];
        pq4_set_packed_element(block, code & 15, bbs, nsq, offset, 2 * i);
        pq4_set_packed_element(block, code >> 4, bbs, nsq, offset, 2 * i + 1);
    }
}

void CodePackerPQ4::unpack_1(
        const uint8_t* block,
        size_t offset,
        uint8_t* flat_code) const {
    size_t bbs = nvec;
    if (offset >= nvec) {
        block += (offset / nvec) * block_size;
        offset = offset % nvec;
    }
    for (size_t i = 0; i < code_size; i++) {
        uint8_t code0, code1;
        code0 = pq4_get_packed_element(block, bbs, nsq, offset, 2 * i);
        code1 = pq4_get_packed_element(block, bbs, nsq, offset, 2 * i + 1);
        flat_code[i] = code0 | (code1 << 4);
    }
}

CodePacker* CodePackerPQ4::clone() const {
    return new CodePackerPQ4(*this);
}

/***************************************************************
 * Packing functions for Look-Up Tables (LUT)
 ***************************************************************/

void pq4_pack_LUT(int nq, int nsq, const uint8_t* src, uint8_t* dest) {
    for (int q = 0; q < nq; q++) {
        for (int sq = 0; sq < nsq; sq += 2) {
            memcpy(dest + (sq / 2 * nq + q) * 32,
                   src + (q * nsq + sq) * 16,
                   16);
            memcpy(dest + (sq / 2 * nq + q) * 32 + 16,
                   src + (q * nsq + sq + 1) * 16,
                   16);
        }
    }
}

int pq4_pack_LUT_qbs(int qbs, int nsq, const uint8_t* src, uint8_t* dest) {
    FAISS_THROW_IF_NOT(nsq % 2 == 0);
    size_t dim12 = 16 * nsq;
    int i0 = 0;
    int qi = qbs;
    while (qi) {
        int nq = qi & 15;
        qi >>= 4;
        pq4_pack_LUT(nq, nsq, src + i0 * dim12, dest + i0 * dim12);
        i0 += nq;
    }
    return i0;
}

namespace {

void pack_LUT_1_q_map(
        int nq,
        const int* q_map,
        int nsq,
        const uint8_t* src,
        uint8_t* dest) {
    for (int qi = 0; qi < nq; qi++) {
        int q = q_map[qi];
        for (int sq = 0; sq < nsq; sq += 2) {
            memcpy(dest + (sq / 2 * nq + qi) * 32,
                   src + (q * nsq + sq) * 16,
                   16);
            memcpy(dest + (sq / 2 * nq + qi) * 32 + 16,
                   src + (q * nsq + sq + 1) * 16,
                   16);
        }
    }
}

} // anonymous namespace

int pq4_pack_LUT_qbs_q_map(
        int qbs,
        int nsq,
        const uint8_t* src,
        const int* q_map,
        uint8_t* dest) {
    FAISS_THROW_IF_NOT(nsq % 2 == 0);
    size_t dim12 = 16 * nsq;
    int i0 = 0;
    int qi = qbs;
    while (qi) {
        int nq = qi & 15;
        qi >>= 4;
        pack_LUT_1_q_map(nq, q_map + i0, nsq, src, dest + i0 * dim12);
        i0 += nq;
    }
    return i0;
}

// declared in simd_result_handlers.h
bool simd_result_handlers_accept_virtual = true;

int pq4_qbs_to_nq(int qbs) {
    int i0 = 0;
    int qi = qbs;
    while (qi) {
        int nq = qi & 15;
        qi >>= 4;
        i0 += nq;
    }
    return i0;
}

int pq4_preferred_qbs(int n) {
    // from timings in P141901742, P141902828
    static int map[12] = {
            0, 1, 2, 3, 0x13, 0x23, 0x33, 0x223, 0x233, 0x333, 0x2233, 0x2333};
    if (n <= 11) {
        return map[n];
    } else if (n <= 24) {
        // override qbs: all first stages with 3 steps
        // then 1 stage with the rest
        int nbit = 4 * (n / 3); // nbits with only 3s
        int qbs = 0x33333333 & ((1 << nbit) - 1);
        qbs |= (n % 3) << nbit;
        return qbs;
    } else {
        FAISS_THROW_FMT("number of queries %d too large", n);
    }
}

void accumulate_to_mem(
        int nq,
        size_t ntotal2,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        uint16_t* accu) {
    using namespace simd_result_handlers;
    FAISS_THROW_IF_NOT(ntotal2 % 32 == 0);
    StoreResultHandler<> handler(accu, ntotal2);
    DummyScaler<> scaler;
    accumulate(nq, ntotal2, nsq, codes, LUT, handler, scaler, 32 * nsq / 2);
}

} // namespace faiss

/***************************************************************
 * FastScanCodeScanner: NONE specialization + dispatch wrapper.
 *
 * The NONE specialization provides the scalar fallback.
 * Per-SIMD specializations (AVX2, AVX512, ARM_NEON) are in
 * impl-avx2.cpp, impl-avx512.cpp, impl-neon.cpp respectively.
 ***************************************************************/

#define THE_LEVEL_TO_DISPATCH SIMDLevel::NONE
#include <faiss/impl/fast_scan/dispatching.h>        // IWYU pragma: keep
#include <faiss/impl/fast_scan/rabitq_dispatching.h> // IWYU pragma: keep
#undef THE_LEVEL_TO_DISPATCH

namespace faiss {

std::unique_ptr<FastScanCodeScanner> make_fast_scan_knn_scanner(
        bool is_max,
        int impl,
        size_t nq,
        size_t ntotal,
        int64_t k,
        float* distances,
        int64_t* ids,
        const IDSelector* sel,
        bool with_id_map) {
    DISPATCH_SIMDLevel(
            make_fast_scan_scanner_impl,
            is_max,
            impl,
            nq,
            ntotal,
            k,
            distances,
            ids,
            sel,
            with_id_map);
}

std::unique_ptr<FastScanCodeScanner> make_range_scanner(
        bool is_max,
        RangeSearchResult& rres,
        float radius,
        size_t ntotal,
        const IDSelector* sel) {
    DISPATCH_SIMDLevel(
            make_range_scanner_impl, is_max, rres, radius, ntotal, sel);
}

std::unique_ptr<FastScanCodeScanner> make_partial_range_scanner(
        bool is_max,
        RangeSearchPartialResult& pres,
        float radius,
        size_t ntotal,
        size_t q0,
        size_t q1,
        const IDSelector* sel) {
    DISPATCH_SIMDLevel(
            make_partial_range_scanner_impl,
            is_max,
            pres,
            radius,
            ntotal,
            q0,
            q1,
            sel);
}

std::unique_ptr<FastScanCodeScanner> rabitq_make_knn_scanner(
        const IndexRaBitQFastScan* index,
        bool is_max,
        size_t nq,
        int64_t k,
        float* distances,
        int64_t* ids,
        const IDSelector* sel,
        const FastScanDistancePostProcessing& context,
        bool is_multi_bit) {
    DISPATCH_SIMDLevel(
            rabitq_make_knn_scanner_impl,
            index,
            is_max,
            nq,
            k,
            distances,
            ids,
            sel,
            context,
            is_multi_bit);
}

std::unique_ptr<FastScanCodeScanner> rabitq_ivf_make_knn_scanner(
        bool is_max,
        const IndexIVFRaBitQFastScan* index,
        size_t nq,
        size_t k,
        float* distances,
        int64_t* ids,
        const FastScanDistancePostProcessing* context,
        bool multi_bit) {
    DISPATCH_SIMDLevel(
            rabitq_ivf_make_knn_scanner_impl,
            is_max,
            index,
            nq,
            k,
            distances,
            ids,
            context,
            multi_bit);
}

} // namespace faiss
