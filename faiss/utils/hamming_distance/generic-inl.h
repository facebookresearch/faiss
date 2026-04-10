/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef HAMMING_GENERIC_INL_H
#define HAMMING_GENERIC_INL_H

// A general-purpose version of hamming distance computation.
// Most code is now in common.h. This file only contains the
// GenHammingComputer classes that use scalar generalized_hamming_64.

#include <cassert>
#include <cstdint>

#include <faiss/impl/platform_macros.h>

namespace faiss {

struct GenHammingComputer16 {
    uint64_t a0, a1;
    GenHammingComputer16(const uint8_t* a8, FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 16);
        const uint64_t* a = reinterpret_cast<const uint64_t*>(a8);
        a0 = a[0];
        a1 = a[1];
    }

    inline int hamming(const uint8_t* b8) const {
        const uint64_t* b = reinterpret_cast<const uint64_t*>(b8);
        return generalized_hamming_64(b[0] ^ a0) +
                generalized_hamming_64(b[1] ^ a1);
    }

    inline static constexpr int get_code_size() {
        return 16;
    }
};

struct GenHammingComputer32 {
    uint64_t a0, a1, a2, a3;

    GenHammingComputer32(const uint8_t* a8, FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 32);
        const uint64_t* a = reinterpret_cast<const uint64_t*>(a8);
        a0 = a[0];
        a1 = a[1];
        a2 = a[2];
        a3 = a[3];
    }

    inline int hamming(const uint8_t* b8) const {
        const uint64_t* b = reinterpret_cast<const uint64_t*>(b8);
        return generalized_hamming_64(b[0] ^ a0) +
                generalized_hamming_64(b[1] ^ a1) +
                generalized_hamming_64(b[2] ^ a2) +
                generalized_hamming_64(b[3] ^ a3);
    }

    inline static constexpr int get_code_size() {
        return 32;
    }
};

struct GenHammingComputerM8 {
    const uint64_t* a;
    int n;

    GenHammingComputerM8(const uint8_t* a8, int code_size) {
        assert(code_size % 8 == 0);
        a = reinterpret_cast<const uint64_t*>(a8);
        n = code_size / 8;
    }

    int hamming(const uint8_t* b8) const {
        const uint64_t* b = reinterpret_cast<const uint64_t*>(b8);
        int accu = 0;
        for (int i = 0; i < n; i++)
            accu += generalized_hamming_64(a[i] ^ b[i]);
        return accu;
    }

    inline int get_code_size() const {
        return n * 8;
    }
};

} // namespace faiss

#endif
