/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Per-ISA implementation of Hamming distance computation for
 * IndexBinaryHNSW. Included once per SIMD TU with THE_SIMD_LEVEL
 * set to the desired SIMDLevel.
 */

#pragma once

#ifndef THE_SIMD_LEVEL
#error "THE_SIMD_LEVEL must be defined before including this file"
#endif

// The including TU (or the per-ISA hamming_computer-*.h it pulls in first)
// is responsible for providing the HammingComputer*_tpl<SL> specializations;
// this header only needs the forward declarations and with_HammingComputer<SL>
// dispatcher from hamming_computer.h.
#include <faiss/utils/hamming_distance/hamming_computer.h>

#include <faiss/IndexBinaryFlat.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/binary_hamming/dispatch.h>
#include <faiss/utils/hamming.h>

namespace faiss {

namespace {

template <class HammingComputer>
struct FlatHammingDis : DistanceComputer {
    const int code_size;
    const uint8_t* b;
    HammingComputer hc;

    float operator()(idx_t i) override {
        return hc.hamming(b + i * code_size);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return HammingComputerDefault_tpl<THE_SIMD_LEVEL>(
                       b + j * code_size, code_size)
                .hamming(b + i * code_size);
    }

    explicit FlatHammingDis(const IndexBinaryFlat& storage)
            : code_size(storage.code_size), b(storage.xb.data()), hc() {}

    // NOTE: Pointers are cast from float in order to reuse the floating-point
    //   DistanceComputer.
    void set_query(const float* x) override {
        hc.set((uint8_t*)x, code_size);
    }
};

} // anonymous namespace

template <>
DistanceComputer* make_binary_hnsw_distance_computer_fixSL<THE_SIMD_LEVEL>(
        int code_size,
        IndexBinaryFlat* flat_storage) {
    return with_HammingComputer<THE_SIMD_LEVEL>(
            code_size, [&]<class HammingComputer>() -> DistanceComputer* {
                return new FlatHammingDis<HammingComputer>(*flat_storage);
            });
}

} // namespace faiss
