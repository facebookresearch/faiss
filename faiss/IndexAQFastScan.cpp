/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexAQFastScan.h>

#include <limits.h>
#include <cassert>
#include <memory>

#include <omp.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/LocalSearchQuantizer.h>
#include <faiss/impl/ResidualQuantizer.h>
#include <faiss/impl/pq4_fast_scan.h>
#include <faiss/utils/utils.h>

namespace faiss {

inline size_t roundup(size_t a, size_t b) {
    return (a + b - 1) / b * b;
}

IndexAQFastScan::IndexAQFastScan(
        AdditiveQuantizer* aq,
        MetricType metric,
        int bbs) {
    init(aq, metric, bbs);
}

void IndexAQFastScan::init(AdditiveQuantizer* aq, MetricType metric, int bbs) {
    FAISS_THROW_IF_NOT(aq != nullptr);
    FAISS_THROW_IF_NOT(!aq->nbits.empty());
    FAISS_THROW_IF_NOT(aq->nbits[0] == 4);
    if (metric == METRIC_INNER_PRODUCT) {
        FAISS_THROW_IF_NOT_MSG(
                aq->search_type == AdditiveQuantizer::ST_LUT_nonorm,
                "Search type must be ST_LUT_nonorm for IP metric");
    } else {
        FAISS_THROW_IF_NOT_MSG(
                aq->search_type == AdditiveQuantizer::ST_norm_lsq2x4 ||
                        aq->search_type == AdditiveQuantizer::ST_norm_rq2x4,
                "Search type must be lsq2x4 or rq2x4 for L2 metric");
    }

    this->aq = aq;
    if (metric == METRIC_L2) {
        M = aq->M + 2; // 2x4 bits AQ
    } else {
        M = aq->M;
    }
    init_fastscan(aq->d, M, 4, metric, bbs);

    max_train_points = 1024 * ksub * M;
}

IndexAQFastScan::IndexAQFastScan() : IndexFastScan() {
    is_trained = false;
    aq = nullptr;
}

IndexAQFastScan::IndexAQFastScan(const IndexAdditiveQuantizer& orig, int bbs) {
    init(orig.aq, orig.metric_type, bbs);

    ntotal = orig.ntotal;
    is_trained = orig.is_trained;
    orig_codes = orig.codes.data();

    ntotal2 = roundup(ntotal, bbs);
    codes.resize(ntotal2 * M2 / 2);
    pq4_pack_codes(orig_codes, ntotal, M, ntotal2, bbs, M2, codes.get());
}

IndexAQFastScan::~IndexAQFastScan() {}

void IndexAQFastScan::train(idx_t n, const float* x_in) {
    if (is_trained) {
        return;
    }

    const int seed = 0x12345;
    size_t nt = n;
    const float* x = fvecs_maybe_subsample(
            d, &nt, max_train_points, x_in, verbose, seed);
    n = nt;
    if (verbose) {
        printf("training additive quantizer on %zd vectors\n", nt);
    }

    aq->verbose = verbose;
    aq->train(n, x);

    is_trained = true;
}

void IndexAQFastScan::compute_codes(uint8_t* tmp_codes, idx_t n, const float* x)
        const {
    aq->compute_codes(x, tmp_codes, n);
}

void IndexAQFastScan::compute_float_LUT(float* lut, idx_t n, const float* x)
        const {
    if (metric_type == METRIC_INNER_PRODUCT) {
        aq->compute_LUT(n, x, lut, 1.0f);
    } else {
        // compute inner product look-up tables
        const size_t ip_dim12 = aq->M * ksub;
        const size_t norm_dim12 = 2 * ksub;
        std::vector<float> ip_lut(n * ip_dim12);
        aq->compute_LUT(n, x, ip_lut.data(), -2.0f);

        // norm look-up tables
        const float* norm_lut = aq->norm_tabs.data();
        FAISS_THROW_IF_NOT(aq->norm_tabs.size() == norm_dim12);

        // combine them
        for (idx_t i = 0; i < n; i++) {
            memcpy(lut, ip_lut.data() + i * ip_dim12, ip_dim12 * sizeof(*lut));
            lut += ip_dim12;
            memcpy(lut, norm_lut, norm_dim12 * sizeof(*lut));
            lut += norm_dim12;
        }
    }
}

/**************************************************************************************
 * IndexRQFastScan
 **************************************************************************************/

IndexRQFastScan::IndexRQFastScan(
        int d,        ///< dimensionality of the input vectors
        size_t M,     ///< number of subquantizers
        size_t nbits, ///< number of bit per subvector index
        MetricType metric,
        Search_type_t search_type,
        int bbs)
        : rq(d, M, nbits, search_type) {
    init(&rq, metric, bbs);
}

IndexRQFastScan::IndexRQFastScan() {
    aq = &rq;
}

/**************************************************************************************
 * IndexLSQFastScan
 **************************************************************************************/

IndexLSQFastScan::IndexLSQFastScan(
        int d,
        size_t M,     ///< number of subquantizers
        size_t nbits, ///< number of bit per subvector index
        MetricType metric,
        Search_type_t search_type,
        int bbs)
        : lsq(d, M, nbits, search_type) {
    init(&lsq, metric, bbs);
}

IndexLSQFastScan::IndexLSQFastScan() {
    aq = &lsq;
}

} // namespace faiss