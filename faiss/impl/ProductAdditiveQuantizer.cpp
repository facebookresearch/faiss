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

#include <faiss/clone_index.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
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

ProductAdditiveQuantizer::ProductAdditiveQuantizer(
        size_t d,
        const std::vector<AdditiveQuantizer*>& aqs,
        Search_type_t search_type) {
    init(d, aqs, search_type);
}

ProductAdditiveQuantizer::ProductAdditiveQuantizer() {}

void ProductAdditiveQuantizer::init(
        size_t d,
        const std::vector<AdditiveQuantizer*>& aqs,
        Search_type_t search_type) {
    FAISS_THROW_IF_NOT_MSG(
            !aqs.empty(), "At least one additive quantizer is required.");
    for (size_t i = 0; i < aqs.size(); i++) {
        const auto& q = aqs[i];
        FAISS_THROW_IF_NOT(q->d == aqs[0]->d);
        FAISS_THROW_IF_NOT(q->M == aqs[0]->M);
        FAISS_THROW_IF_NOT(q->nbits[0] == aqs[0]->nbits[0]);
    }

    // AdditiveQuantizer constructor
    this->d = d;
    this->search_type = search_type;
    M = aqs.size() * aqs[0]->M;
    nbits = std::vector<size_t>(M, aqs[0]->nbits[0]);
    verbose = false;
    is_trained = false;
    norm_max = norm_min = NAN;
    code_size = 0;
    tot_bits = 0;
    total_codebook_size = 0;
    only_8bit = false;
    set_derived_values();

    // ProductAdditiveQuantizer
    nsplits = aqs.size();

    FAISS_THROW_IF_NOT(quantizers.empty());
    for (const auto& q : aqs) {
        auto aq = dynamic_cast<AdditiveQuantizer*>(clone_Quantizer(q));
        quantizers.push_back(aq);
    }
}

ProductAdditiveQuantizer::~ProductAdditiveQuantizer() {
    for (auto& q : quantizers) {
        delete q;
    }
}

AdditiveQuantizer* ProductAdditiveQuantizer::subquantizer(size_t s) const {
    return quantizers[s];
}

void ProductAdditiveQuantizer::train(size_t n, const float* x) {
    if (is_trained) {
        return;
    }

    // copy the subvectors into contiguous memory
    size_t offset_d = 0;
    std::vector<float> xt;
    for (size_t s = 0; s < nsplits; s++) {
        auto q = quantizers[s];
        xt.resize(q->d * n);

#pragma omp parallel for if (n > 1000)
        for (idx_t i = 0; i < n; i++) {
            memcpy(xt.data() + i * q->d,
                   x + i * d + offset_d,
                   q->d * sizeof(*x));
        }

        q->train(n, xt.data());
        offset_d += q->d;
    }

    // compute codebook size
    size_t codebook_size = 0;
    for (const auto& q : quantizers) {
        codebook_size += q->total_codebook_size * q->d;
    }

    // copy codebook from sub-quantizers
    codebooks.resize(codebook_size); // size (M * ksub, dsub)
    float* cb = codebooks.data();
    for (size_t s = 0; s < nsplits; s++) {
        auto q = quantizers[s];
        size_t sub_codebook_size = q->total_codebook_size * q->d;
        memcpy(cb, q->codebooks.data(), sub_codebook_size * sizeof(float));
        cb += sub_codebook_size;
    }

    is_trained = true;
}

void ProductAdditiveQuantizer::compute_codes_add_centroids(
        const float* x,
        uint8_t* codes_out,
        size_t n,
        const float* centroids) const {
    // size (n, M)
    std::vector<int32_t> unpacked_codes(n * M);

    /// TODO: actuallly we do not need to unpack and pack
    size_t offset_d = 0, offset_m = 0;
    std::vector<float> xsub;
    std::vector<uint8_t> codes;

    for (size_t s = 0; s < nsplits; s++) {
        const auto q = quantizers[s];
        xsub.resize(n * q->d);
        codes.resize(n * q->code_size);

#pragma omp parallel for if (n > 1000)
        for (idx_t i = 0; i < n; i++) {
            memcpy(xsub.data() + i * q->d,
                   x + i * d + offset_d,
                   q->d * sizeof(float));
        }

        q->compute_codes(xsub.data(), codes.data(), n);

        // unpack
#pragma omp parallel for if (n > 1000)
        for (idx_t i = 0; i < n; i++) {
            uint8_t* code = codes.data() + i * q->code_size;
            BitstringReader bsr(code, q->code_size);

            // unpacked_codes[i][s][m] = codes[i][m]
            for (size_t m = 0; m < q->M; m++) {
                unpacked_codes[i * M + offset_m + m] = bsr.read(q->nbits[m]);
            }
        }

        offset_d += q->d;
        offset_m += q->M;
    }

    // pack
    pack_codes(n, unpacked_codes.data(), codes_out, -1, nullptr, centroids);
}

void ProductAdditiveQuantizer::decode_unpacked(
        const int32_t* codes,
        float* x,
        size_t n,
        int64_t ld_codes) const {
    FAISS_THROW_IF_NOT_MSG(
            is_trained, "The product additive quantizer is not trained yet.");

    if (ld_codes == -1) {
        ld_codes = M;
    }

    // product additive quantizer decoding
#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        const int32_t* codesi = codes + i * ld_codes;

        size_t offset_m = 0, offset_d = 0;
        for (size_t s = 0; s < nsplits; s++) {
            const auto q = quantizers[s];
            float* xi = x + i * d + offset_d;

            for (int m = 0; m < q->M; m++) {
                int idx = codesi[offset_m + m];
                const float* c = codebooks.data() +
                        q->d * (codebook_offsets[offset_m + m] + idx);
                if (m == 0) {
                    memcpy(xi, c, sizeof(*x) * q->d);
                } else {
                    fvec_add(q->d, xi, c, xi);
                }
            }

            offset_m += q->M;
            offset_d += q->d;
        }
    }
}

void ProductAdditiveQuantizer::decode(const uint8_t* codes, float* x, size_t n)
        const {
    FAISS_THROW_IF_NOT_MSG(
            is_trained, "The product additive quantizer is not trained yet.");

#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        BitstringReader bsr(codes + i * code_size, code_size);

        size_t offset_m = 0, offset_d = 0;
        for (size_t s = 0; s < nsplits; s++) {
            const auto q = quantizers[s];
            float* xi = x + i * d + offset_d;

            for (int m = 0; m < q->M; m++) {
                int idx = bsr.read(q->nbits[m]);
                const float* c = codebooks.data() +
                        q->d * (codebook_offsets[offset_m + m] + idx);
                if (m == 0) {
                    memcpy(xi, c, sizeof(*x) * q->d);
                } else {
                    fvec_add(q->d, xi, c, xi);
                }
            }

            offset_m += q->M;
            offset_d += q->d;
        }
    }
}

void ProductAdditiveQuantizer::compute_LUT(
        size_t n,
        const float* xq,
        float* LUT,
        float alpha,
        long ld_lut) const {
    // codebooks:  size (M * ksub, dsub)
    // xq:         size (n, d)
    // output LUT: size (n, M * ksub)

    FINTEGER nqi = n;
    // leading dimension of 'LUT' and 'xq'
    FINTEGER ld_LUT = ld_lut > 0 ? ld_lut : total_codebook_size;
    FINTEGER ld_xq = d;

    float zero = 0;
    size_t offset_d = 0;
    size_t offset_cb = 0;
    size_t offset_lut = 0;

    for (size_t s = 0; s < nsplits; s++) {
        const auto q = quantizers[s];

        FINTEGER ncenti = q->total_codebook_size;
        FINTEGER ld_cb = q->d; // leading dimension of 'codebooks'

        auto codebooksi = codebooks.data() + offset_cb;
        auto xqi = xq + offset_d;
        auto LUTi = LUT + offset_lut;

        sgemm_("Transposed",
               "Not transposed",
               &ncenti,
               &nqi,
               &ld_cb,
               &alpha,
               codebooksi,
               &ld_cb,
               xqi,
               &ld_xq,
               &zero,
               LUTi,
               &ld_LUT);

        offset_d += q->d;
        offset_cb += q->total_codebook_size * q->d;
        offset_lut += q->total_codebook_size;
    }
}

/*************************************
 * Product Local Search Quantizer
 ************************************/

ProductLocalSearchQuantizer::ProductLocalSearchQuantizer(
        size_t d,
        size_t nsplits,
        size_t Msub,
        size_t nbits,
        Search_type_t search_type) {
    FAISS_THROW_IF_NOT(d % nsplits == 0);
    size_t dsub = d / nsplits;
    std::vector<AdditiveQuantizer*> aqs;

    for (size_t i = 0; i < nsplits; i++) {
        auto lsq = new LocalSearchQuantizer(dsub, Msub, nbits, ST_decompress);
        aqs.push_back(lsq);
    }
    init(d, aqs, search_type);

    for (auto& q : aqs) {
        delete q;
    }
}

ProductLocalSearchQuantizer::ProductLocalSearchQuantizer() {}

/*************************************
 * Product Residual Quantizer
 ************************************/

ProductResidualQuantizer::ProductResidualQuantizer(
        size_t d,
        size_t nsplits,
        size_t Msub,
        size_t nbits,
        Search_type_t search_type) {
    FAISS_THROW_IF_NOT(d % nsplits == 0);
    size_t dsub = d / nsplits;
    std::vector<AdditiveQuantizer*> aqs;

    for (size_t i = 0; i < nsplits; i++) {
        auto rq = new ResidualQuantizer(dsub, Msub, nbits, ST_decompress);
        aqs.push_back(rq);
    }
    init(d, aqs, search_type);

    for (auto& q : aqs) {
        delete q;
    }
}

ProductResidualQuantizer::ProductResidualQuantizer() {}

} // namespace faiss
