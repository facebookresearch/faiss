/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/impl/AdditiveQuantizer.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>

#include <algorithm>

#include <faiss/Clustering.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/LocalSearchQuantizer.h>
#include <faiss/impl/ResidualQuantizer.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h>
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

namespace faiss {

AdditiveQuantizer::AdditiveQuantizer(
        size_t d,
        const std::vector<size_t>& nbits,
        Search_type_t search_type)
        : Quantizer(d),
          M(nbits.size()),
          nbits(nbits),
          search_type(search_type) {
    set_derived_values();
}

AdditiveQuantizer::AdditiveQuantizer()
        : AdditiveQuantizer(0, std::vector<size_t>()) {}

void AdditiveQuantizer::set_derived_values() {
    tot_bits = 0;
    only_8bit = true;
    codebook_offsets.resize(M + 1, 0);
    for (int i = 0; i < M; i++) {
        int nbit = nbits[i];
        size_t k = 1 << nbit;
        codebook_offsets[i + 1] = codebook_offsets[i] + k;
        tot_bits += nbit;
        if (nbit != 0) {
            only_8bit = false;
        }
    }
    total_codebook_size = codebook_offsets[M];
    switch (search_type) {
        case ST_norm_float:
            norm_bits = 32;
            break;
        case ST_norm_qint8:
        case ST_norm_cqint8:
        case ST_norm_lsq2x4:
        case ST_norm_rq2x4:
            norm_bits = 8;
            break;
        case ST_norm_qint4:
        case ST_norm_cqint4:
            norm_bits = 4;
            break;
        case ST_decompress:
        case ST_LUT_nonorm:
        case ST_norm_from_LUT:
        default:
            norm_bits = 0;
            break;
    }
    tot_bits += norm_bits;

    // convert bits to bytes
    code_size = (tot_bits + 7) / 8;
}

void AdditiveQuantizer::train_norm(size_t n, const float* norms) {
    norm_min = HUGE_VALF;
    norm_max = -HUGE_VALF;
    for (idx_t i = 0; i < n; i++) {
        if (norms[i] < norm_min) {
            norm_min = norms[i];
        }
        if (norms[i] > norm_max) {
            norm_max = norms[i];
        }
    }

    if (search_type == ST_norm_cqint8 || search_type == ST_norm_cqint4) {
        size_t k = (1 << 8);
        if (search_type == ST_norm_cqint4) {
            k = (1 << 4);
        }
        Clustering1D clus(k);
        clus.train_exact(n, norms);
        qnorm.add(clus.k, clus.centroids.data());
    } else if (search_type == ST_norm_lsq2x4 || search_type == ST_norm_rq2x4) {
        std::unique_ptr<AdditiveQuantizer> aq;
        if (search_type == ST_norm_lsq2x4) {
            aq.reset(new LocalSearchQuantizer(1, 2, 4));
        } else {
            aq.reset(new ResidualQuantizer(1, 2, 4));
        }

        aq->train(n, norms);
        // flatten aq codebooks
        std::vector<float> flat_codebooks(1 << 8);
        FAISS_THROW_IF_NOT(aq->codebooks.size() == 32);

        // save norm tables for 4-bit fastscan search
        norm_tabs = aq->codebooks;

        // assume big endian
        const float* c = norm_tabs.data();
        for (size_t i = 0; i < 16; i++) {
            for (size_t j = 0; j < 16; j++) {
                flat_codebooks[i * 16 + j] = c[j] + c[16 + i];
            }
        }

        qnorm.reset();
        qnorm.add(1 << 8, flat_codebooks.data());
        FAISS_THROW_IF_NOT(qnorm.ntotal == (1 << 8));
    }
}

void AdditiveQuantizer::compute_codebook_tables() {
    centroid_norms.resize(total_codebook_size);
    fvec_norms_L2sqr(
            centroid_norms.data(), codebooks.data(), d, total_codebook_size);
    size_t cross_table_size = 0;
    for (int m = 0; m < M; m++) {
        size_t K = (size_t)1 << nbits[m];
        cross_table_size += K * codebook_offsets[m];
    }
    codebook_cross_products.resize(cross_table_size);
    size_t ofs = 0;
    for (int m = 1; m < M; m++) {
        FINTEGER ki = (size_t)1 << nbits[m];
        FINTEGER kk = codebook_offsets[m];
        FINTEGER di = d;
        float zero = 0, one = 1;
        assert(ofs + ki * kk <= cross_table_size);
        sgemm_("Transposed",
               "Not transposed",
               &ki,
               &kk,
               &di,
               &one,
               codebooks.data() + d * kk,
               &di,
               codebooks.data(),
               &di,
               &zero,
               codebook_cross_products.data() + ofs,
               &ki);
        ofs += ki * kk;
    }
}

namespace {

// TODO
// https://stackoverflow.com/questions/31631224/hacks-for-clamping-integer-to-0-255-and-doubles-to-0-0-1-0

uint8_t encode_qint8(float x, float amin, float amax) {
    float x1 = (x - amin) / (amax - amin) * 256;
    int32_t xi = int32_t(floor(x1));

    return xi < 0 ? 0 : xi > 255 ? 255 : xi;
}

uint8_t encode_qint4(float x, float amin, float amax) {
    float x1 = (x - amin) / (amax - amin) * 16;
    int32_t xi = int32_t(floor(x1));

    return xi < 0 ? 0 : xi > 15 ? 15 : xi;
}

float decode_qint8(uint8_t i, float amin, float amax) {
    return (i + 0.5) / 256 * (amax - amin) + amin;
}

float decode_qint4(uint8_t i, float amin, float amax) {
    return (i + 0.5) / 16 * (amax - amin) + amin;
}

} // anonymous namespace

uint32_t AdditiveQuantizer::encode_qcint(float x) const {
    idx_t id;
    qnorm.assign(1, &x, &id, 1);
    return uint32_t(id);
}

float AdditiveQuantizer::decode_qcint(uint32_t c) const {
    return qnorm.get_xb()[c];
}

uint64_t AdditiveQuantizer::encode_norm(float norm) const {
    switch (search_type) {
        case ST_norm_float:
            uint32_t inorm;
            memcpy(&inorm, &norm, 4);
            return inorm;
        case ST_norm_qint8:
            return encode_qint8(norm, norm_min, norm_max);
        case ST_norm_qint4:
            return encode_qint4(norm, norm_min, norm_max);
        case ST_norm_lsq2x4:
        case ST_norm_rq2x4:
        case ST_norm_cqint8:
            return encode_qcint(norm);
        case ST_norm_cqint4:
            return encode_qcint(norm);
        case ST_decompress:
        case ST_LUT_nonorm:
        case ST_norm_from_LUT:
        default:
            return 0;
    }
}

void AdditiveQuantizer::pack_codes(
        size_t n,
        const int32_t* codes,
        uint8_t* packed_codes,
        int64_t ld_codes,
        const float* norms,
        const float* centroids) const {
    if (ld_codes == -1) {
        ld_codes = M;
    }
    std::vector<float> norm_buf;
    if (search_type == ST_norm_float || search_type == ST_norm_qint4 ||
        search_type == ST_norm_qint8 || search_type == ST_norm_cqint8 ||
        search_type == ST_norm_cqint4 || search_type == ST_norm_lsq2x4 ||
        search_type == ST_norm_rq2x4) {
        if (centroids != nullptr || !norms) {
            norm_buf.resize(n);
            std::vector<float> x_recons(n * d);
            decode_unpacked(codes, x_recons.data(), n, ld_codes);

            if (centroids != nullptr) {
                // x = x + c
                fvec_add(n * d, x_recons.data(), centroids, x_recons.data());
            }
            fvec_norms_L2sqr(norm_buf.data(), x_recons.data(), d, n);
            norms = norm_buf.data();
        }
    }
#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        const int32_t* codes1 = codes + i * ld_codes;
        BitstringWriter bsw(packed_codes + i * code_size, code_size);
        for (int m = 0; m < M; m++) {
            bsw.write(codes1[m], nbits[m]);
        }
        if (norm_bits != 0) {
            bsw.write(encode_norm(norms[i]), norm_bits);
        }
    }
}

void AdditiveQuantizer::decode(const uint8_t* code, float* x, size_t n) const {
    FAISS_THROW_IF_NOT_MSG(
            is_trained, "The additive quantizer is not trained yet.");

    // standard additive quantizer decoding
#pragma omp parallel for if (n > 100)
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

void AdditiveQuantizer::decode_unpacked(
        const int32_t* code,
        float* x,
        size_t n,
        int64_t ld_codes) const {
    FAISS_THROW_IF_NOT_MSG(
            is_trained, "The additive quantizer is not trained yet.");

    if (ld_codes == -1) {
        ld_codes = M;
    }

    // standard additive quantizer decoding
#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        const int32_t* codesi = code + i * ld_codes;
        float* xi = x + i * d;
        for (int m = 0; m < M; m++) {
            int idx = codesi[m];
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
 * Support for fast distance computations in centroids
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

void AdditiveQuantizer::compute_LUT(
        size_t n,
        const float* xq,
        float* LUT,
        float alpha,
        long ld_lut) const {
    // in all cases, it is large matrix multiplication

    FINTEGER ncenti = total_codebook_size;
    FINTEGER di = d;
    FINTEGER nqi = n;
    FINTEGER ldc = ld_lut > 0 ? ld_lut : ncenti;
    float zero = 0;

    sgemm_("Transposed",
           "Not transposed",
           &ncenti,
           &nqi,
           &di,
           &alpha,
           codebooks.data(),
           &di,
           xq,
           &di,
           &zero,
           LUT,
           &ldc);
}

namespace {

/* compute inner products of one query with all centroids, given a look-up
 * table of all inner producst with codebook entries */
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

void AdditiveQuantizer::knn_centroids_inner_product(
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

void AdditiveQuantizer::knn_centroids_L2(
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

/****************************************************************************
 * Support for fast distance computations in codes
 ****************************************************************************/

namespace {

float accumulate_IPs(
        const AdditiveQuantizer& aq,
        BitstringReader& bs,
        const float* LUT) {
    float accu = 0;
    for (int m = 0; m < aq.M; m++) {
        size_t nbit = aq.nbits[m];
        int idx = bs.read(nbit);
        accu += LUT[idx];
        LUT += (uint64_t)1 << nbit;
    }
    return accu;
}

float compute_norm_from_LUT(const AdditiveQuantizer& aq, BitstringReader& bs) {
    float accu = 0;
    std::vector<int> idx(aq.M);
    const float* c = aq.codebook_cross_products.data();
    for (int m = 0; m < aq.M; m++) {
        size_t nbit = aq.nbits[m];
        int i = bs.read(nbit);
        size_t K = 1 << nbit;
        idx[m] = i;

        accu += aq.centroid_norms[aq.codebook_offsets[m] + i];

        for (int l = 0; l < m; l++) {
            int j = idx[l];
            accu += 2 * c[j * K + i];
            c += (1 << aq.nbits[l]) * K;
        }
    }
    // FAISS_THROW_IF_NOT(c == aq.codebook_cross_products.data() +
    // aq.codebook_cross_products.size());
    return accu;
}

} // anonymous namespace

template <>
float AdditiveQuantizer::
        compute_1_distance_LUT<true, AdditiveQuantizer::ST_LUT_nonorm>(
                const uint8_t* codes,
                const float* LUT) const {
    BitstringReader bs(codes, code_size);
    return accumulate_IPs(*this, bs, LUT);
}

template <>
float AdditiveQuantizer::
        compute_1_distance_LUT<false, AdditiveQuantizer::ST_LUT_nonorm>(
                const uint8_t* codes,
                const float* LUT) const {
    BitstringReader bs(codes, code_size);
    return -accumulate_IPs(*this, bs, LUT);
}

template <>
float AdditiveQuantizer::
        compute_1_distance_LUT<false, AdditiveQuantizer::ST_norm_float>(
                const uint8_t* codes,
                const float* LUT) const {
    BitstringReader bs(codes, code_size);
    float accu = accumulate_IPs(*this, bs, LUT);
    uint32_t norm_i = bs.read(32);
    float norm2;
    memcpy(&norm2, &norm_i, 4);
    return norm2 - 2 * accu;
}

template <>
float AdditiveQuantizer::
        compute_1_distance_LUT<false, AdditiveQuantizer::ST_norm_cqint8>(
                const uint8_t* codes,
                const float* LUT) const {
    BitstringReader bs(codes, code_size);
    float accu = accumulate_IPs(*this, bs, LUT);
    uint32_t norm_i = bs.read(8);
    float norm2 = decode_qcint(norm_i);
    return norm2 - 2 * accu;
}

template <>
float AdditiveQuantizer::
        compute_1_distance_LUT<false, AdditiveQuantizer::ST_norm_cqint4>(
                const uint8_t* codes,
                const float* LUT) const {
    BitstringReader bs(codes, code_size);
    float accu = accumulate_IPs(*this, bs, LUT);
    uint32_t norm_i = bs.read(4);
    float norm2 = decode_qcint(norm_i);
    return norm2 - 2 * accu;
}

template <>
float AdditiveQuantizer::
        compute_1_distance_LUT<false, AdditiveQuantizer::ST_norm_qint8>(
                const uint8_t* codes,
                const float* LUT) const {
    BitstringReader bs(codes, code_size);
    float accu = accumulate_IPs(*this, bs, LUT);
    uint32_t norm_i = bs.read(8);
    float norm2 = decode_qint8(norm_i, norm_min, norm_max);
    return norm2 - 2 * accu;
}

template <>
float AdditiveQuantizer::
        compute_1_distance_LUT<false, AdditiveQuantizer::ST_norm_qint4>(
                const uint8_t* codes,
                const float* LUT) const {
    BitstringReader bs(codes, code_size);
    float accu = accumulate_IPs(*this, bs, LUT);
    uint32_t norm_i = bs.read(4);
    float norm2 = decode_qint4(norm_i, norm_min, norm_max);
    return norm2 - 2 * accu;
}

template <>
float AdditiveQuantizer::
        compute_1_distance_LUT<false, AdditiveQuantizer::ST_norm_from_LUT>(
                const uint8_t* codes,
                const float* LUT) const {
    FAISS_THROW_IF_NOT(codebook_cross_products.size() > 0);
    BitstringReader bs(codes, code_size);
    float accu = accumulate_IPs(*this, bs, LUT);
    BitstringReader bs2(codes, code_size);
    float norm2 = compute_norm_from_LUT(*this, bs2);
    return norm2 - 2 * accu;
}

} // namespace faiss
