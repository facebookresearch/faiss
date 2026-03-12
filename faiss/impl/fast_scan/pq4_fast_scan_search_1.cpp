/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/fast_scan/fast_scan.h>

#include <faiss/impl/fast_scan/accumulate_loops.h>
#include <faiss/impl/fast_scan/simd_result_handlers.h>

namespace faiss {

using namespace simd_result_handlers;

/***************************************************************
 * accumulation functions
 ***************************************************************/

namespace {

template <class ResultHandler>
void pq4_accumulate_loop_fixed_handler(
        int nq,
        size_t nb,
        int bbs,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        int pq2x4_scale,
        size_t block_stride) {
    if (pq2x4_scale) {
        NormTableScaler<> scaler(pq2x4_scale);
        pq4_accumulate_loop_fixed_scaler(
                nq, nb, bbs, nsq, codes, LUT, res, scaler, block_stride);
    } else {
        DummyScaler<> dscaler;
        pq4_accumulate_loop_fixed_scaler(
                nq, nb, bbs, nsq, codes, LUT, res, dscaler, block_stride);
    }
}

} // anonymous namespace

void pq4_accumulate_loop(
        int nq,
        size_t nb,
        int bbs,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        SIMDResultHandler& res,
        int pq2x4_scale,
        size_t block_stride) {
    with_SIMDResultHandler(res, [&](auto& handler) {
        pq4_accumulate_loop_fixed_handler(
                nq,
                nb,
                bbs,
                nsq,
                codes,
                LUT,
                handler,
                pq2x4_scale,
                block_stride);
    });
}

} // namespace faiss
