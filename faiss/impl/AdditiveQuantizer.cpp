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

#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h> // BitstringWriter
#include <faiss/utils/utils.h>

extern "C" {

// general matrix multiplication
int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);
}

namespace {

// c and a and b can overlap
void fvec_add(size_t d, const float* a, const float* b, float* c) {
    for (size_t i = 0; i < d; i++) {
        c[i] = a[i] + b[i];
    }
}

void fvec_add(size_t d, const float* a, float b, float* c) {
    for (size_t i = 0; i < d; i++) {
        c[i] = a[i] + b;
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
    total_codebook_size = codebook_offsets[M];
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

/****************************************************************************
 * Support for fast distance computations and search with additive quantizer
 ****************************************************************************/

void AdditiveQuantizer::compute_centroid_norms(float* norms) const {
    size_t ntotal = (size_t)1 << tot_bits;
    // TODO: make tree of partial sums
#pragma omp parallel
    {
        std::vector<float> tmp(d);
#pragma omp for
        for (int64_t i = 0; i < ntotal; i++) {
            decode_64bit(i, tmp.data());
            norms[i] = fvec_norm_L2sqr(tmp.data(), d);
        }
    }
}

void AdditiveQuantizer::decode_64bit(idx_t bits, float* xi) const {
    for (int m = 0; m < M; m++) {
        idx_t idx = bits & (((size_t)1 << nbits[m]) - 1);
        bits >>= nbits[m];
        const float* c = codebooks.data() + d * (codebook_offsets[m] + idx);
        if (m == 0) {
            memcpy(xi, c, sizeof(*xi) * d);
        } else {
            fvec_add(d, xi, c, xi);
        }
    }
}

void AdditiveQuantizer::compute_LUT(size_t n, const float* xq, float* LUT)
        const {
    // in all cases, it is large matrix multiplication

    FINTEGER ncenti = total_codebook_size;
    FINTEGER di = d;
    FINTEGER nqi = n;
    float one = 1, zero = 0;

    sgemm_("Transposed",
           "Not transposed",
           &ncenti,
           &nqi,
           &di,
           &one,
           codebooks.data(),
           &di,
           xq,
           &di,
           &zero,
           LUT,
           &ncenti);
}

namespace {

void compute_inner_prod_with_LUT(
        const AdditiveQuantizer& aq,
        const float* LUT,
        float* ips) {
    size_t prev_size = 1;
    for (int m = 0; m < aq.M; m++) {
        const float* LUTm = LUT + aq.codebook_offsets[m];
        int nb = aq.nbits[m];
        size_t nc = (size_t)1 << nb;

        if (m == 0) {
            memcpy(ips, LUT, sizeof(*ips) * nc);
        } else {
            for (int64_t i = nc - 1; i >= 0; i--) {
                float v = LUTm[i];
                fvec_add(prev_size, ips, v, ips + i * prev_size);
            }
        }
        prev_size *= nc;
    }
}

} // anonymous namespace

void AdditiveQuantizer::knn_exact_inner_product(
        idx_t n,
        const float* xq,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    std::unique_ptr<float[]> LUT(new float[n * total_codebook_size]);
    compute_LUT(n, xq, LUT.get());
    size_t ntotal = (size_t)1 << tot_bits;

#pragma omp parallel if (n > 100)
    {
        std::vector<float> dis(ntotal);
#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            const float* LUTi = LUT.get() + i * total_codebook_size;
            compute_inner_prod_with_LUT(*this, LUTi, dis.data());
            float* distances_i = distances + i * k;
            idx_t* labels_i = labels + i * k;
            minheap_heapify(k, distances_i, labels_i);
            minheap_addn(k, distances_i, labels_i, dis.data(), nullptr, ntotal);
            minheap_reorder(k, distances_i, labels_i);
        }
    }
}

void AdditiveQuantizer::knn_exact_L2(
        idx_t n,
        const float* xq,
        idx_t k,
        float* distances,
        idx_t* labels,
        const float* norms) const {
    std::unique_ptr<float[]> LUT(new float[n * total_codebook_size]);
    compute_LUT(n, xq, LUT.get());
    std::unique_ptr<float[]> q_norms(new float[n]);
    fvec_norms_L2sqr(q_norms.get(), xq, d, n);
    size_t ntotal = (size_t)1 << tot_bits;

#pragma omp parallel if (n > 100)
    {
        std::vector<float> dis(ntotal);
#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            const float* LUTi = LUT.get() + i * total_codebook_size;
            float* distances_i = distances + i * k;
            idx_t* labels_i = labels + i * k;

            compute_inner_prod_with_LUT(*this, LUTi, dis.data());

            // update distances using
            // ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * <x,y>

            maxheap_heapify(k, distances_i, labels_i);
            for (idx_t j = 0; j < ntotal; j++) {
                float disj = q_norms[i] + norms[j] - 2 * dis[j];
                if (disj < distances_i[0]) {
                    heap_replace_top<CMax<float, int64_t>>(
                            k, distances_i, labels_i, disj, j);
                }
            }
            maxheap_reorder(k, distances_i, labels_i);
        }
    }
}

} // namespace faiss
