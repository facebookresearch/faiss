/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexIVFAdditiveQuantizerFastScan.h>

#include <cassert>
#include <cinttypes>
#include <cstdio>

#include <omp.h>

#include <memory>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/LookupTableScaler.h>
#include <faiss/impl/pq4_fast_scan.h>
#include <faiss/invlists/BlockInvertedLists.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/quantize_lut.h>
#include <faiss/utils/simdlib.h>
#include <faiss/utils/utils.h>

namespace faiss {

inline size_t roundup(size_t a, size_t b) {
    return (a + b - 1) / b * b;
}

IndexIVFAdditiveQuantizerFastScan::IndexIVFAdditiveQuantizerFastScan(
        Index* quantizer,
        AdditiveQuantizer* aq,
        size_t d,
        size_t nlist,
        MetricType metric,
        int bbs)
        : IndexIVFFastScan(quantizer, d, nlist, 0, metric) {
    if (aq != nullptr) {
        init(aq, nlist, metric, bbs);
    }
}

void IndexIVFAdditiveQuantizerFastScan::init(
        AdditiveQuantizer* aq,
        size_t nlist,
        MetricType metric,
        int bbs) {
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
    if (metric_type == METRIC_L2) {
        M = aq->M + 2; // 2x4 bits AQ
    } else {
        M = aq->M;
    }
    init_fastscan(M, 4, nlist, metric, bbs);

    max_train_points = 1024 * ksub * M;
    by_residual = true;
}

IndexIVFAdditiveQuantizerFastScan::IndexIVFAdditiveQuantizerFastScan(
        const IndexIVFAdditiveQuantizer& orig,
        int bbs)
        : IndexIVFFastScan(
                  orig.quantizer,
                  orig.d,
                  orig.nlist,
                  0,
                  orig.metric_type),
          aq(orig.aq) {
    FAISS_THROW_IF_NOT(
            metric_type == METRIC_INNER_PRODUCT || !orig.by_residual);

    init(aq, nlist, metric_type, bbs);

    is_trained = orig.is_trained;
    ntotal = orig.ntotal;
    nprobe = orig.nprobe;

    for (size_t i = 0; i < nlist; i++) {
        size_t nb = orig.invlists->list_size(i);
        size_t nb2 = roundup(nb, bbs);
        AlignedTable<uint8_t> tmp(nb2 * M2 / 2);
        pq4_pack_codes(
                InvertedLists::ScopedCodes(orig.invlists, i).get(),
                nb,
                M,
                nb2,
                bbs,
                M2,
                tmp.get());
        invlists->add_entries(
                i,
                nb,
                InvertedLists::ScopedIds(orig.invlists, i).get(),
                tmp.get());
    }

    orig_invlists = orig.invlists;
}

IndexIVFAdditiveQuantizerFastScan::IndexIVFAdditiveQuantizerFastScan() {
    bbs = 0;
    M2 = 0;
    aq = nullptr;

    is_trained = false;
}

IndexIVFAdditiveQuantizerFastScan::~IndexIVFAdditiveQuantizerFastScan() {}

/*********************************************************
 * Training
 *********************************************************/

void IndexIVFAdditiveQuantizerFastScan::train_residual(
        idx_t n,
        const float* x_in) {
    if (aq->is_trained) {
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

    std::unique_ptr<float[]> del_x;
    if (x != x_in) {
        del_x.reset((float*)x);
    }

    const float* trainset;
    std::vector<float> residuals(n * d);
    std::vector<idx_t> assign(n);

    if (by_residual) {
        if (verbose) {
            printf("computing residuals\n");
        }
        quantizer->assign(n, x, assign.data());
        residuals.resize(n * d);
        for (idx_t i = 0; i < n; i++) {
            quantizer->compute_residual(
                    x + i * d, residuals.data() + i * d, assign[i]);
        }
        trainset = residuals.data();
    } else {
        trainset = x;
    }

    if (verbose) {
        printf("training %zdx%zd additive quantizer on "
               "%" PRId64 " vectors in %dD\n",
               aq->M,
               ksub,
               n,
               d);
    }
    aq->verbose = verbose;
    aq->train(n, trainset);

    // train norm quantizer
    if (by_residual && metric_type == METRIC_L2) {
        std::vector<float> decoded_x(n * d);
        std::vector<uint8_t> x_codes(n * aq->code_size);
        aq->compute_codes(residuals.data(), x_codes.data(), n);
        aq->decode(x_codes.data(), decoded_x.data(), n);

        // add coarse centroids
        FAISS_THROW_IF_NOT(assign.size() == n);
        std::vector<float> centroid(d);
        for (idx_t i = 0; i < n; i++) {
            auto xi = decoded_x.data() + i * d;
            quantizer->reconstruct(assign[i], centroid.data());
            fvec_add(d, centroid.data(), xi, xi);
        }

        std::vector<float> norms(n, 0);
        fvec_norms_L2sqr(norms.data(), decoded_x.data(), d, n);

        // re-train norm tables
        aq->train_norm(n, norms.data());
    }

    if (metric_type == METRIC_L2) {
        estimate_norm_scale(n, x);
    }
}

void IndexIVFAdditiveQuantizerFastScan::estimate_norm_scale(
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

    std::vector<idx_t> coarse_ids(n);
    std::vector<float> coarse_dis(n);
    quantizer->search(n, x, 1, coarse_dis.data(), coarse_ids.data());

    AlignedTable<float> dis_tables;
    AlignedTable<float> biases;

    size_t index_nprobe = nprobe;
    nprobe = 1;
    compute_LUT(n, x, coarse_ids.data(), coarse_dis.data(), dis_tables, biases);
    nprobe = index_nprobe;

    float scale = 0;

#pragma omp parallel for reduction(+ : scale)
    for (idx_t i = 0; i < n; i++) {
        const float* lut = dis_tables.get() + i * M * ksub;
        scale += quantize_lut::aq_estimate_norm_scale(M, ksub, 2, lut);
    }
    scale /= n;
    norm_scale = (int)std::roundf(std::max(scale, 1.0f));

    /// TODO: if verbose
    printf("estimated norm scale: %lf\n", scale);
    printf("rounded norm scale: %d\n", norm_scale);
}

/*********************************************************
 * Code management functions
 *********************************************************/

void IndexIVFAdditiveQuantizerFastScan::encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listnos) const {
    idx_t bs = 65536;
    if (n > bs) {
        for (idx_t i0 = 0; i0 < n; i0 += bs) {
            idx_t i1 = std::min(n, i0 + bs);
            encode_vectors(
                    i1 - i0,
                    x + i0 * d,
                    list_nos + i0,
                    codes + i0 * code_size,
                    include_listnos);
        }
        return;
    }

    if (by_residual) {
        std::vector<float> residuals(n * d);
        std::vector<float> centroids(n * d);

#pragma omp parallel for if (n > 1000)
        for (idx_t i = 0; i < n; i++) {
            if (list_nos[i] < 0) {
                memset(residuals.data() + i * d, 0, sizeof(residuals[0]) * d);
            } else {
                quantizer->compute_residual(
                        x + i * d, residuals.data() + i * d, list_nos[i]);
            }
        }

#pragma omp parallel for if (n > 1000)
        for (idx_t i = 0; i < n; i++) {
            auto c = centroids.data() + i * d;
            quantizer->reconstruct(list_nos[i], c);
        }

        aq->compute_codes(residuals.data(), codes, n, centroids.data());
    } else {
        aq->compute_codes(x, codes, n);
    }

    if (include_listnos) {
        size_t coarse_size = coarse_code_size();
        for (idx_t i = n - 1; i >= 0; i--) {
            uint8_t* code = codes + i * (coarse_size + code_size);
            memmove(code + coarse_size, codes + i * code_size, code_size);
            encode_listno(list_nos[i], code);
        }
    }
}

/*********************************************************
 * Search functions
 *********************************************************/

void IndexIVFAdditiveQuantizerFastScan::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    FAISS_THROW_IF_NOT(k > 0);
    bool rescale = (rescale_norm && norm_scale > 1 && metric_type == METRIC_L2);
    if (!rescale) {
        IndexIVFFastScan::search(n, x, k, distances, labels);
        return;
    }

    NormTableScaler scaler(norm_scale, M - 2);
    if (metric_type == METRIC_L2) {
        search_dispatch_implem<true>(n, x, k, distances, labels, scaler);
    } else {
        search_dispatch_implem<false>(n, x, k, distances, labels, scaler);
    }
}

/*********************************************************
 * Look-Up Table functions
 *********************************************************/

/**
 * d(x, y) = || x - (y_c + \sum_i y_i) ||^2
 *         = || x ||^2 - 2 <x, y_c> - 2 \sum_i<x, y_i> + || y ||^2
 */

bool IndexIVFAdditiveQuantizerFastScan::lookup_table_is_3d() const {
    return false;
}

void IndexIVFAdditiveQuantizerFastScan::compute_LUT(
        size_t n,
        const float* x,
        const idx_t* coarse_ids,
        const float* coarse_dis,
        AlignedTable<float>& dis_tables,
        AlignedTable<float>& biases) const {
    const size_t dim12 = ksub * M;
    const size_t ip_dim12 = aq->M * ksub;

    dis_tables.resize(n * dim12);

    float coef = 1.0f;
    if (metric_type == METRIC_L2) {
        coef = -2.0f;
    }

    if (by_residual) {
        // coef * <x, y_c>
        biases.resize(n * nprobe);
#pragma omp parallel
        {
            std::vector<float> centroid(d);
            float* c = centroid.data();

#pragma omp for
            for (idx_t ij = 0; ij < n * nprobe; ij++) {
                int i = ij / nprobe;
                quantizer->reconstruct(coarse_ids[ij], c);
                biases[ij] = coef * fvec_inner_product(c, x + i * d, d);
            }
        }
    }

    if (metric_type == METRIC_L2) {
        const size_t norm_dim12 = 2 * ksub;

        // inner product look-up tables
        aq->compute_LUT(n, x, dis_tables.data(), -2.0f, dim12);

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
#pragma omp parallel for if (n > 100)
        for (idx_t i = 0; i < n; i++) {
            float* tab = dis_tables.data() + i * dim12 + ip_dim12;
            memcpy(tab, norm_lut, norm_dim12 * sizeof(*tab));
        }

    } else if (metric_type == METRIC_INNER_PRODUCT) {
        aq->compute_LUT(n, x, dis_tables.get());
    } else {
        FAISS_THROW_FMT("metric %d not supported", metric_type);
    }
}

void IndexIVFAdditiveQuantizerFastScan::sa_decode(
        idx_t n,
        const uint8_t* bytes,
        float* x) const {
    aq->decode(bytes, x, n);
}

/********** IndexIVFLocalSearchQuantizerFastScan ************/
IndexIVFLocalSearchQuantizerFastScan::IndexIVFLocalSearchQuantizerFastScan(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t M,
        size_t nbits,
        MetricType metric,
        Search_type_t search_type,
        int bbs)
        : IndexIVFAdditiveQuantizerFastScan(
                  quantizer,
                  nullptr,
                  d,
                  nlist,
                  metric,
                  bbs),
          lsq(d, M, nbits, search_type) {
    FAISS_THROW_IF_NOT(nbits == 4); // TODO: delete me
    init(&lsq, nlist, metric, bbs);
}

IndexIVFLocalSearchQuantizerFastScan::IndexIVFLocalSearchQuantizerFastScan() {
    aq = &lsq;
}

IndexIVFLocalSearchQuantizerFastScan::~IndexIVFLocalSearchQuantizerFastScan() {}

/********** IndexIVFResidualQuantizerFastScan ************/
IndexIVFResidualQuantizerFastScan::IndexIVFResidualQuantizerFastScan(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t M,
        size_t nbits,
        MetricType metric,
        Search_type_t search_type,
        int bbs)
        : IndexIVFAdditiveQuantizerFastScan(
                  quantizer,
                  nullptr,
                  d,
                  nlist,
                  metric,
                  bbs),
          rq(d, M, nbits, search_type) {
    FAISS_THROW_IF_NOT(nbits == 4); // TODO: delete me
    init(&rq, nlist, metric, bbs);
}

IndexIVFResidualQuantizerFastScan::IndexIVFResidualQuantizerFastScan() {
    aq = &rq;
}

IndexIVFResidualQuantizerFastScan::~IndexIVFResidualQuantizerFastScan() {}

} // namespace faiss