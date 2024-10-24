/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexAdditiveQuantizerFastScan.h>

#include <cassert>
#include <climits>
#include <memory>

#include <omp.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/LocalSearchQuantizer.h>
#include <faiss/impl/LookupTableScaler.h>
#include <faiss/impl/ResidualQuantizer.h>
#include <faiss/impl/pq4_fast_scan.h>
#include <faiss/utils/quantize_lut.h>
#include <faiss/utils/utils.h>

namespace faiss {

inline size_t roundup(size_t a, size_t b) {
    return (a + b - 1) / b * b;
}

IndexAdditiveQuantizerFastScan::IndexAdditiveQuantizerFastScan(
        AdditiveQuantizer* aq,
        MetricType metric,
        int bbs) {
    init(aq, metric, bbs);
}

void IndexAdditiveQuantizerFastScan::init(
        AdditiveQuantizer* aq_2,
        MetricType metric,
        int bbs) {
    FAISS_THROW_IF_NOT(aq_2 != nullptr);
    FAISS_THROW_IF_NOT(!aq_2->nbits.empty());
    FAISS_THROW_IF_NOT(aq_2->nbits[0] == 4);
    if (metric == METRIC_INNER_PRODUCT) {
        FAISS_THROW_IF_NOT_MSG(
                aq_2->search_type == AdditiveQuantizer::ST_LUT_nonorm,
                "Search type must be ST_LUT_nonorm for IP metric");
    } else {
        FAISS_THROW_IF_NOT_MSG(
                aq_2->search_type == AdditiveQuantizer::ST_norm_lsq2x4 ||
                        aq_2->search_type == AdditiveQuantizer::ST_norm_rq2x4,
                "Search type must be lsq2x4 or rq2x4 for L2 metric");
    }

    this->aq = aq_2;
    if (metric == METRIC_L2) {
        M = aq_2->M + 2; // 2x4 bits AQ
    } else {
        M = aq_2->M;
    }
    init_fastscan(aq_2->d, M, 4, metric, bbs);

    max_train_points = 1024 * ksub * M;
}

IndexAdditiveQuantizerFastScan::IndexAdditiveQuantizerFastScan()
        : IndexFastScan() {
    is_trained = false;
    aq = nullptr;
}

IndexAdditiveQuantizerFastScan::IndexAdditiveQuantizerFastScan(
        const IndexAdditiveQuantizer& orig,
        int bbs) {
    init(orig.aq, orig.metric_type, bbs);

    ntotal = orig.ntotal;
    is_trained = orig.is_trained;
    orig_codes = orig.codes.data();

    ntotal2 = roundup(ntotal, bbs);
    codes.resize(ntotal2 * M2 / 2);
    pq4_pack_codes(orig_codes, ntotal, M, ntotal2, bbs, M2, codes.get());
}

IndexAdditiveQuantizerFastScan::~IndexAdditiveQuantizerFastScan() = default;

void IndexAdditiveQuantizerFastScan::train(idx_t n, const float* x_in) {
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
    if (metric_type == METRIC_L2) {
        estimate_norm_scale(n, x);
    }

    is_trained = true;
}

void IndexAdditiveQuantizerFastScan::estimate_norm_scale(
        idx_t n,
        const float* x_in) {
    FAISS_THROW_IF_NOT(metric_type == METRIC_L2);

    constexpr int seed = 0x980903;
    constexpr size_t max_points_estimated = 65536;
    size_t ns = n;
    const float* x = fvecs_maybe_subsample(
            d, &ns, max_points_estimated, x_in, verbose, seed);
    n = ns;
    std::unique_ptr<float[]> del_x;
    if (x != x_in) {
        del_x.reset((float*)x);
    }

    std::vector<float> dis_tables(n * M * ksub);
    compute_float_LUT(dis_tables.data(), n, x);

    // here we compute the mean of scales for each query
    // TODO: try max of scales
    double scale = 0;

#pragma omp parallel for reduction(+ : scale)
    for (idx_t i = 0; i < n; i++) {
        const float* lut = dis_tables.data() + i * M * ksub;
        scale += quantize_lut::aq_estimate_norm_scale(M, ksub, 2, lut);
    }
    scale /= n;
    norm_scale = (int)std::roundf(std::max(scale, 1.0));

    if (verbose) {
        printf("estimated norm scale: %lf\n", scale);
        printf("rounded norm scale: %d\n", norm_scale);
    }
}

void IndexAdditiveQuantizerFastScan::compute_codes(
        uint8_t* tmp_codes,
        idx_t n,
        const float* x) const {
    aq->compute_codes(x, tmp_codes, n);
}

void IndexAdditiveQuantizerFastScan::compute_float_LUT(
        float* lut,
        idx_t n,
        const float* x) const {
    if (metric_type == METRIC_INNER_PRODUCT) {
        aq->compute_LUT(n, x, lut, 1.0f);
    } else {
        // compute inner product look-up tables
        const size_t ip_dim12 = aq->M * ksub;
        const size_t norm_dim12 = 2 * ksub;
        std::vector<float> ip_lut(n * ip_dim12);
        aq->compute_LUT(n, x, ip_lut.data(), -2.0f);

        // copy and rescale norm look-up tables
        auto norm_tabs = aq->norm_tabs;
        if (rescale_norm && norm_scale > 1 && metric_type == METRIC_L2) {
            for (size_t i = 0; i < norm_tabs.size(); i++) {
                norm_tabs[i] /= norm_scale;
            }
        }
        const float* norm_lut = norm_tabs.data();
        FAISS_THROW_IF_NOT(norm_tabs.size() == norm_dim12);

        // combine them
        for (idx_t i = 0; i < n; i++) {
            memcpy(lut, ip_lut.data() + i * ip_dim12, ip_dim12 * sizeof(*lut));
            lut += ip_dim12;
            memcpy(lut, norm_lut, norm_dim12 * sizeof(*lut));
            lut += norm_dim12;
        }
    }
}

void IndexAdditiveQuantizerFastScan::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");
    FAISS_THROW_IF_NOT(k > 0);
    bool rescale = (rescale_norm && norm_scale > 1 && metric_type == METRIC_L2);
    if (!rescale) {
        IndexFastScan::search(n, x, k, distances, labels);
        return;
    }

    NormTableScaler scaler(norm_scale);
    if (metric_type == METRIC_L2) {
        search_dispatch_implem<true>(n, x, k, distances, labels, &scaler);
    } else {
        search_dispatch_implem<false>(n, x, k, distances, labels, &scaler);
    }
}

void IndexAdditiveQuantizerFastScan::sa_decode(
        idx_t n,
        const uint8_t* bytes,
        float* x) const {
    aq->decode(bytes, x, n);
}

/**************************************************************************************
 * IndexResidualQuantizerFastScan
 **************************************************************************************/

IndexResidualQuantizerFastScan::IndexResidualQuantizerFastScan(
        int d,        ///< dimensionality of the input vectors
        size_t M,     ///< number of subquantizers
        size_t nbits, ///< number of bit per subvector index
        MetricType metric,
        Search_type_t search_type,
        int bbs)
        : rq(d, M, nbits, search_type) {
    init(&rq, metric, bbs);
}

IndexResidualQuantizerFastScan::IndexResidualQuantizerFastScan() {
    aq = &rq;
}

/**************************************************************************************
 * IndexLocalSearchQuantizerFastScan
 **************************************************************************************/

IndexLocalSearchQuantizerFastScan::IndexLocalSearchQuantizerFastScan(
        int d,
        size_t M,     ///< number of subquantizers
        size_t nbits, ///< number of bit per subvector index
        MetricType metric,
        Search_type_t search_type,
        int bbs)
        : lsq(d, M, nbits, search_type) {
    init(&lsq, metric, bbs);
}

IndexLocalSearchQuantizerFastScan::IndexLocalSearchQuantizerFastScan() {
    aq = &lsq;
}

/**************************************************************************************
 * IndexProductResidualQuantizerFastScan
 **************************************************************************************/

IndexProductResidualQuantizerFastScan::IndexProductResidualQuantizerFastScan(
        int d,          ///< dimensionality of the input vectors
        size_t nsplits, ///< number of residual quantizers
        size_t Msub,    ///< number of subquantizers per RQ
        size_t nbits,   ///< number of bit per subvector index
        MetricType metric,
        Search_type_t search_type,
        int bbs)
        : prq(d, nsplits, Msub, nbits, search_type) {
    init(&prq, metric, bbs);
}

IndexProductResidualQuantizerFastScan::IndexProductResidualQuantizerFastScan() {
    aq = &prq;
}

/**************************************************************************************
 * IndexProductLocalSearchQuantizerFastScan
 **************************************************************************************/

IndexProductLocalSearchQuantizerFastScan::
        IndexProductLocalSearchQuantizerFastScan(
                int d,          ///< dimensionality of the input vectors
                size_t nsplits, ///< number of local search quantizers
                size_t Msub,    ///< number of subquantizers per LSQ
                size_t nbits,   ///< number of bit per subvector index
                MetricType metric,
                Search_type_t search_type,
                int bbs)
        : plsq(d, nsplits, Msub, nbits, search_type) {
    init(&plsq, metric, bbs);
}

IndexProductLocalSearchQuantizerFastScan::
        IndexProductLocalSearchQuantizerFastScan() {
    aq = &plsq;
}

} // namespace faiss
