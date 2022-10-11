/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/pq4_fast_scan.h>
#include <faiss/impl/simd_result_handlers.h>

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
        uint8_t* blocks) {
    FAISS_THROW_IF_NOT(bbs % 32 == 0);
    FAISS_THROW_IF_NOT(nb % bbs == 0);
    FAISS_THROW_IF_NOT(nsq % 2 == 0);

    memset(blocks, 0, nb * nsq / 2);
    const uint8_t perm0[16] = {
            0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};

    uint8_t* codes2 = blocks;
    for (size_t i0 = 0; i0 < nb; i0 += bbs) {
        for (int sq = 0; sq < nsq; sq += 2) {
            for (size_t i = 0; i < bbs; i += 32) {
                std::array<uint8_t, 32> c, c0, c1;
                get_matrix_column(
                        codes, ntotal, (M + 1) / 2, i0 + i, sq / 2, c);
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
        size_t M2,
        uint8_t* blocks) {
    const uint8_t perm0[16] = {
            0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};

    // range of affected blocks
    size_t block0 = i0 / bbs;
    size_t block1 = ((i1 - 1) / bbs) + 1;

    for (size_t b = block0; b < block1; b++) {
        uint8_t* codes2 = blocks + b * bbs * M2 / 2;
        int64_t i_base = b * bbs - i0;
        for (int sq = 0; sq < M2; sq += 2) {
            for (size_t i = 0; i < bbs; i += 32) {
                std::array<uint8_t, 32> c, c0, c1;
                get_matrix_column(
                        codes, i1 - i0, (M + 1) / 2, i_base + i, sq / 2, c);
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
uint8_t get_vector_specific_address(
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

} // namespace faiss
