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

// hamdis-inl.h MUST be included first — the ISA ladder checks __AVX2__ etc.
// which are set by per-ISA TU compilation flags.
#include <faiss/utils/hamming_distance/hamdis-inl.h>

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
        return HammingComputerDefault(b + j * code_size, code_size)
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

struct BuildDistanceComputer {
    using T = DistanceComputer*;
    template <class HammingComputer>
    DistanceComputer* f(IndexBinaryFlat* flat_storage) {
        return new FlatHammingDis<HammingComputer>(*flat_storage);
    }
};

} // anonymous namespace

template <>
DistanceComputer* make_binary_hnsw_distance_computer_dispatch<THE_SIMD_LEVEL>(
        int code_size,
        IndexBinaryFlat* flat_storage) {
    BuildDistanceComputer bd;
    return dispatch_HammingComputer(code_size, bd, flat_storage);
}

} // namespace faiss
