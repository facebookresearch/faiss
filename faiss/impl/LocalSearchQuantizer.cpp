/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/LocalSearchQuantizer.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>

#include <algorithm>

#include <faiss/IndexFlat.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/utils.h>

namespace faiss {

LocalSearchQuantizer::LocalSearchQuantizer(size_t d, size_t M, size_t nbits) {
    FAISS_THROW_IF_NOT(nbits == 8);
    this->d = d;
    this->M = M;
    this->nbits = nbits;

    verbose = false;
    code_size = M * (nbits / 8); // in bytes

    train_iters = 25;
    encode_iters = 25;
    ils_iters = 8;
    icm_iters = 4;
}

namespace {

void fvec_sub(size_t d, const float* a, const float* b, float* c) {
    for (size_t i = 0; i < d; i++) {
        c[i] = a[i] - b[i];
    }
}

// c and a and b can overlap
void fvec_add(size_t d, const float* a, const float* b, float* c) {
    for (size_t i = 0; i < d; i++) {
        c[i] = a[i] + b[i];
    }
}

void random_int32(std::vector<int32_t> *x, int32_t min,
        int32_t max, std::mt19937 &gen) {
    std::uniform_int_distribution<int32_t> distrib(min, max);
    for (size_t i = 0; i < x->size(); i++) {
        (*x)[i] = distrib(gen);
    }
}

void random_float(std::vector<float> *x, float min,
        float max, std::mt19937 &gen) {
    std::uniform_real_distribution<float> distrib(min, max);
    for (size_t i = 0; i < x->size(); i++) {
        (*x)[i] = distrib(gen);
    }
}

} // anonymous namespace


void LocalSearchQuantizer::train(size_t n, const float* x) {
    if (verbose) {
        printf("Training LocalSearchQuantizer, with %zd subcodes on %zd %zdD vectors\n",
               M,
               n,
               size_t(d));
    }

    size_t h = (1 << nbits);           // number of codes per codebook
    codebooks.resize(M * h * d);       // [M, h, d]
    std::vector<int32_t> codes(n * M); // [n, M]

    std::mt19937 gen(12345);
    random_int32(&codes, 0, h - 1, gen);

    // TODO: add SR-D
    // cov = np.diag(np.cov(x.T))
    // mean = np.zeros((d,))

    float obj = evaluate(codebooks.data(), codes.data(), x, n);
    printf("Init obj: %lf\n", obj);

}

void LocalSearchQuantizer::compute_codes(
        const float* x,
        uint8_t* codes_out,
        size_t n) const {
    FAISS_THROW_MSG("Not implemented yet!");
}


void LocalSearchQuantizer::pack_codes(
        size_t n,
        const int32_t* codes,
        uint8_t* packed_codes,
        int64_t ld_codes) const {
    FAISS_THROW_MSG("Not implemented yet!");
}

void LocalSearchQuantizer::decode(const uint8_t* code, float* x, size_t n) const {
    FAISS_THROW_MSG("Not implemented yet!");
}


float LocalSearchQuantizer::evaluate(const float *codebooks, const int32_t *codes,
        const float *x, size_t n) const {
    // decode
    size_t h = (1 << nbits);
    std::vector<float> decoded_x(n * d, 0.0f);
    float obj = 0.0f;

#pragma omp parallel for reduction(+ : obj)
    for (size_t i = 0; i < n; i++) {
        const auto code = codes + i * M;
        const auto decoded_i = decoded_x.data() + i * d;
        for (size_t m = 0; m < M; m++) {
            const auto c = codebooks + m * h * d + code[m] * d; // codebooks[m, code[m]]
            fvec_add(d, decoded_i, c, decoded_i);
        }

        float err = fvec_L2sqr(x + i * d, decoded_i, d);
        obj += err;
    }

    obj = obj / n;
    return obj;
}

} // namespace faiss
