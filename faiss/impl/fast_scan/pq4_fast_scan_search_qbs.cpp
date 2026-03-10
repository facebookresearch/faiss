/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/fast_scan/pq4_fast_scan.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/fast_scan/LookupTableScaler.h>
#include <faiss/impl/fast_scan/decompose_qbs.h>
#include <faiss/impl/fast_scan/simd_result_handlers.h>

namespace faiss {

// declared in simd_result_handlers.h
bool simd_result_handlers_accept_virtual = true;

using namespace simd_result_handlers;

namespace {} // namespace

void pq4_accumulate_loop_qbs(
        int qbs,
        size_t nb,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        SIMDResultHandler& res,
        int pq2x4_scale,
        size_t block_stride) {
    with_SIMDResultHandler(res, [&](auto& handler) {
        if (pq2x4_scale) {
            NormTableScaler<> scaler(pq2x4_scale);
            pq4_accumulate_loop_qbs_fixed_scaler(
                    qbs, nb, nsq, codes, LUT, handler, scaler, block_stride);
        } else {
            DummyScaler<> dummy;
            pq4_accumulate_loop_qbs_fixed_scaler(
                    qbs, nb, nsq, codes, LUT, handler, dummy, block_stride);
        }
    });
}

/***************************************************************
 * Packing functions
 ***************************************************************/

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

void accumulate_to_mem(
        int nq,
        size_t ntotal2,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        uint16_t* accu) {
    FAISS_THROW_IF_NOT(ntotal2 % 32 == 0);
    StoreResultHandler<> handler(accu, ntotal2);
    DummyScaler<> scaler;
    accumulate(nq, ntotal2, nsq, codes, LUT, handler, scaler, 32 * nsq / 2);
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

} // namespace faiss
