/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/pq4_fast_scan.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/LookupTableScaler.h>
#include <faiss/impl/pq_4bit/kernels_simd256.h>
#include <faiss/impl/simd_result_handlers.h>

namespace faiss {

using namespace simd_result_handlers;

/***************************************************************
 * accumulation functions
 ***************************************************************/

namespace {

template <int NQ, int BB, class ResultHandler, class Scaler>
void accumulate_fixed_blocks(
        size_t nb,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler,
        size_t block_stride) {
    constexpr int bbs = 32 * BB;
    for (size_t j0 = 0; j0 < nb; j0 += bbs) {
        FixedStorageHandler<NQ, 2 * BB> res2;
        kernel_accumulate_block<NQ, BB>(nsq, codes, LUT, res2, scaler);
        res.set_block_origin(0, j0);
        res2.to_other_handler(res);
        codes += block_stride;
    }
}

template <class ResultHandler, class Scaler>
void pq4_accumulate_loop_fixed_scaler(
        int nq,
        size_t nb,
        int bbs,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler,
        size_t block_stride) {
    FAISS_THROW_IF_NOT(is_aligned_pointer(codes));
    FAISS_THROW_IF_NOT(is_aligned_pointer(LUT));
    FAISS_THROW_IF_NOT(bbs % 32 == 0);
    FAISS_THROW_IF_NOT(nb % bbs == 0);

#define DISPATCH(NQ, BB)                                         \
    case NQ * 1000 + BB:                                         \
        accumulate_fixed_blocks<NQ, BB>(                         \
                nb, nsq, codes, LUT, res, scaler, block_stride); \
        break

    switch (nq * 1000 + bbs / 32) {
        DISPATCH(1, 1);
        DISPATCH(1, 2);
        DISPATCH(1, 3);
        DISPATCH(1, 4);
        DISPATCH(1, 5);
        DISPATCH(2, 1);
        DISPATCH(2, 2);
        DISPATCH(3, 1);
        DISPATCH(4, 1);
        default:
            FAISS_THROW_FMT("nq=%d bbs=%d not instantiated", nq, bbs);
    }
#undef DISPATCH
}

template <class ResultHandler>
void pq4_accumulate_loop_fixed_handler(
        int nq,
        size_t nb,
        int bbs,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const NormTableScaler* scaler,
        size_t block_stride) {
    if (scaler) {
        pq4_accumulate_loop_fixed_scaler(
                nq, nb, bbs, nsq, codes, LUT, res, *scaler, block_stride);
    } else {
        DummyScaler<> dscaler;
        pq4_accumulate_loop_fixed_scaler(
                nq, nb, bbs, nsq, codes, LUT, res, dscaler, block_stride);
    }
}

struct Run_pq4_accumulate_loop {
    template <class ResultHandler>
    void f(ResultHandler& res,
           int nq,
           size_t nb,
           int bbs,
           int nsq,
           const uint8_t* codes,
           const uint8_t* LUT,
           const NormTableScaler* scaler,
           size_t block_stride) {
        pq4_accumulate_loop_fixed_handler(
                nq, nb, bbs, nsq, codes, LUT, res, scaler, block_stride);
    }
};

} // anonymous namespace

void pq4_accumulate_loop(
        int nq,
        size_t nb,
        int bbs,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        SIMDResultHandler& res,
        const NormTableScaler* scaler,
        size_t block_stride) {
    Run_pq4_accumulate_loop consumer;
    dispatch_SIMDResultHandler(
            res, consumer, nq, nb, bbs, nsq, codes, LUT, scaler, block_stride);
}

} // namespace faiss
