/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/impl/AdditiveQuantizer.h>
#include <faiss/impl/FaissAssert.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>

#include <algorithm>

#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h> // BitstringWriter
#include <faiss/utils/utils.h>

namespace {

// c and a and b can overlap
void fvec_add(size_t d, const float* a, const float* b, float* c) {
    for (size_t i = 0; i < d; i++) {
        c[i] = a[i] + b[i];
    }
}

} // namespace

namespace faiss {

void AdditiveQuantizer::set_derived_values() {
    tot_bits = 0;
    is_byte_aligned = true;
    codebook_offsets.resize(M + 1, 0);
    for (int i = 0; i < M; i++) {
        int nbit = nbits[i];
        size_t k = 1 << nbit;
        codebook_offsets[i + 1] = codebook_offsets[i] + k;
        tot_bits += nbit;
        if (nbit % 8 != 0) {
            is_byte_aligned = false;
        }
    }
    // convert bits to bytes
    code_size = (tot_bits + 7) / 8;
}

void AdditiveQuantizer::pack_codes(
        size_t n,
        const int32_t* codes,
        uint8_t* packed_codes,
        int64_t ld_codes) const {
    if (ld_codes == -1) {
        ld_codes = M;
    }
#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        const int32_t* codes1 = codes + i * ld_codes;
        BitstringWriter bsw(packed_codes + i * code_size, code_size);
        for (int m = 0; m < M; m++) {
            bsw.write(codes1[m], nbits[m]);
        }
    }
}

void AdditiveQuantizer::decode(const uint8_t* code, float* x, size_t n) const {
    FAISS_THROW_IF_NOT_MSG(
            is_trained, "The additive quantizer is not trained yet.");

    // standard additive quantizer decoding
#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        BitstringReader bsr(code + i * code_size, code_size);
        float* xi = x + i * d;
        for (int m = 0; m < M; m++) {
            int idx = bsr.read(nbits[m]);
            const float* c = codebooks.data() + d * (codebook_offsets[m] + idx);
            if (m == 0) {
                memcpy(xi, c, sizeof(*x) * d);
            } else {
                fvec_add(d, xi, c, xi);
            }
        }
    }
}

AdditiveQuantizer::~AdditiveQuantizer() {}

} // namespace faiss
