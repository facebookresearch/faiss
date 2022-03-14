/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/ProductAdditiveQuantizer.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>

#include <algorithm>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/utils.h>

namespace faiss {

ProductAdditiveQuantizer::ProductAdditiveQuantizer(
        size_t d,
        const std::vector<AdditiveQuantizer*>& aqs,
        Search_type_t search_type) {
    init(d, aqs, search_type);
}

void ProductAdditiveQuantizer::init(
        size_t d,
        const std::vector<AdditiveQuantizer*>& aqs,
        Search_type_t search_type) {
    FAISS_THROW_IF_NOT(!aqs.empty());
    for (const auto q : aqs) {
        FAISS_THROW_IF_NOT(q->d == qs[0]->d);
        FAISS_THROW_IF_NOT(q->M == aqs[0]->M);
        FAISS_THROW_IF_NOT(q->nbits[0] == aqs[0]->nbits[0]);
    }

    // AdditiveQuantizer constructor
    this->d = d;
    this->search_type = search_type;
    M = aqs.size() * aqs[0]->M;  // TODO: uneven M. accumulate them
    nbits = std::vector<size_t>(M, aqs[0]->nbits[0]);
    verbose = false;
    is_trained = false;
    norm_max = norm_min = NAN;
    code_size = 0;
    tot_bits = 0;
    total_codebook_size = 0;
    only_8bit = false;
    set_derived_values();

    Msub = aqs[0]->M;
    dsub = qs[0]->d;
    nsplits = aqs.size();
    quantizers = aqs;
}

ProductAdditiveQuantizer::ProductAdditiveQuantizer() {}

void ProductAdditiveQuantizer::train(size_t n, const float* x) {
    if (is_trained) {
        return;
    }

    set_verbose();

    if (M == 1) {
        // only one additive quantizer
        quantizers[0]->train(n, x);
    } else {
        // otherwise we have to copy the subvectors into contiguous space
        size_t d0 = 0;
        std::vector<float> xt(dsub * n);
        for (size_t s = 0; s < nsplits; s++) {
            auto q = quantizers[s];

#pragma omp parallel for
            for (size_t i = 0; i < n; i++) {
                memcpy(xt.data() + i * dsub,
                    x + i * d + d0,
                    dsub * sizeof(float));
            }

            q->train(n, xt.data());
            d0 += dsub;
        }
    }

    // copy codebook from sub-quantizers
    codebooks.resize(total_codebook_size * dsub);  // size (nsplits, Msub, ksub, dsub)
    float* cb = codebooks.data();
    for (size_t s = 0; s < nsplits; s++) {
        auto q = quantizers[s];
        size_t sub_codebook_size = q->total_codebook_size * dsub;  // Msub * ksub * dsub
        memcpy(cb, q->codebooks.data(), sub_codebook_size * sizeof(float));
        cb += sub_codebook_size;
    }

    is_trained = true;
}


void ProductAdditiveQuantizer::compute_codes(const float* x, uint8_t* codes, size_t n) const {
    // only one additive quantizer
    if (nsplits == 1) {
        quantizers[0]->compute_codes(x, codes, n);
        return;
    }

    std::vector<std::vector<uint8_t>> subcodes(nsplits);
    std::vector<std::vector<int32_t>> unpacked_codes(nsplits);

    size_t d0 = 0;
    std::vector<float> xsub(dsub * n);
    for (size_t s = 0; s < nsplits; s++) {
        const auto q = quantizers[s];

#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            memcpy(xsub.data() + i * dsub,
                   x + i * d + d0,
                   dsub * sizeof(float));
        }

        subcodes[s].resize(n * q->code_size);
        q->compute_codes(xsub.data(), subcodes[s].data(), n);
        d0 += dsub;

        // unpack
        unpacked_codes[s].resize(n * q->M);
        
    }

#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        uint8_t* code = codes + i * code_size;
        BitstringWriter bsw(code, code_size);

        for (size_t s = 0; s < nsplits; s++) {
            const auto q = quantizers[s];
            uint8_t* subcode = subcodes[s].data() + i * q->code_size;
            BitstringReader bsr(subcode, q->code_size);

            for (auto b : q->nbits) {
                bsw.write(c, bsr.read(b));
            }
        }
    }
}

void ProductAdditiveQuantizer::decode(const uint8_t* codes, float* x, size_t n) const override;


void ProductAdditiveQuantizer::set_verbose() {
    for (size_t s = 0; s < nsplits; s++) {
        quantizers[s]->verbose = verbose;
    }
}

}  // namespace faiss

