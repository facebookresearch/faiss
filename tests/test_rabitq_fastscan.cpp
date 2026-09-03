/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstring>

#include <faiss/IndexRaBitQFastScan.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/utils/Heap.h>

namespace {

void set_aux_factor(
        faiss::IndexRaBitQFastScan& index,
        size_t vec_pos,
        float distance) {
    const size_t packed_block_size = ((index.M2 + 1) / 2) * index.bbs;
    auto* ptr = faiss::rabitq_utils::get_block_aux_ptr(
            index.codes.get(),
            vec_pos,
            index.bbs,
            packed_block_size,
            index.get_block_stride(),
            index.compute_per_vector_storage_size());

    faiss::SignBitFactors factors;
    factors.or_minus_c_l2sqr = distance;
    factors.dp_multiplier = 0.0f;
    memcpy(ptr, &factors, sizeof(factors));
}

} // namespace

TEST(RaBitQFastScan, HeapHandlerUsesBbsLocalAuxOffset) {
    faiss::IndexRaBitQFastScan index(4, faiss::METRIC_L2, 64, 1);
    index.is_trained = true;
    index.ntotal = 64;
    index.ntotal2 = 64;
    index.codes.resize(index.get_block_stride());
    memset(index.codes.get(), 0, index.codes.size());

    for (size_t i = 0; i < 64; i++) {
        set_aux_factor(index, i, 100.0f);
    }

    // If b=1 incorrectly reads aux offsets 0..31, lane 1 wins with label 33.
    // The correct bbs-local offsets 32..63 make lane 0 win with label 32.
    set_aux_factor(index, 0, 1000.0f);
    set_aux_factor(index, 1, 0.0f);
    set_aux_factor(index, 32, 1.0f);

    float distances[1];
    int64_t labels[1];
    faiss::FastScanDistancePostProcessing context;
    faiss::RaBitQHeapHandler<faiss::CMax<float, int64_t>> handler(
            &index,
            /*nq_val=*/1,
            /*k_val=*/1,
            distances,
            labels,
            /*sel_in=*/nullptr,
            &context,
            /*multi_bit=*/false);

    handler.set_block_origin(/*i0_in=*/0, /*j0_in=*/0);

    using Simd16 = faiss::simd16uint16_tpl<faiss::SINGLE_SIMD_LEVEL_256>;
    Simd16 zero(0);
    handler.handle(/*q=*/0, /*b=*/1, zero, zero);
    handler.end();

    EXPECT_EQ(labels[0], 32);
    EXPECT_FLOAT_EQ(distances[0], 1.0f);
}
