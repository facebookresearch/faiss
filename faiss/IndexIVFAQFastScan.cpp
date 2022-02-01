/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexIVFAQFastScan.h>

#include <cassert>
#include <cinttypes>
#include <cstdio>

#include <omp.h>

#include <memory>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/aq4_fast_scan.h>
#include <faiss/impl/pq4_fast_scan.h>
#include <faiss/impl/simd_result_handlers.h>
#include <faiss/invlists/BlockInvertedLists.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/quantize_lut.h>
#include <faiss/utils/simdlib.h>
#include <faiss/utils/utils.h>

namespace faiss {

using namespace simd_result_handlers;

inline size_t roundup(size_t a, size_t b) {
    return (a + b - 1) / b * b;
}

IndexIVFAQFastScan::IndexIVFAQFastScan(
        Index* quantizer,
        AdditiveQuantizer* aq,
        size_t d,
        size_t nlist,
        MetricType metric,
        int bbs)
        : IndexIVF(quantizer, d, nlist, 0, metric),
          aq(aq),
          bbs(bbs),
          nbits(4),
          ksub(1 << 4) {
    FAISS_THROW_IF_NOT(bbs % 32 == 0);
    FAISS_THROW_IF_NOT(aq != nullptr);
    FAISS_THROW_IF_NOT(!aq->nbits.empty());
    FAISS_THROW_IF_NOT(aq->nbits[0] == 4);
    if (metric == METRIC_INNER_PRODUCT) {
        FAISS_THROW_IF_NOT_MSG(
                aq->search_type == AdditiveQuantizer::ST_decompress,
                "Search type must be ST_decompress for IP metric");
    } else {
        FAISS_THROW_IF_NOT_MSG(
                aq->search_type == AdditiveQuantizer::ST_norm_lsq2x4 ||
                        aq->search_type == AdditiveQuantizer::ST_norm_rq2x4,
                "Search type must be lsq2x4 or rq2x4 for L2 metric");
    }

    M = aq->M + 2;
    M2 = roundup(M, 2);

    is_trained = false;
    code_size = M2 / 2;
    max_training_points = 1024 * ksub * M;

    replace_invlists(new BlockInvertedLists(nlist, bbs, bbs * M2 / 2), true);
}

IndexIVFAQFastScan::IndexIVFAQFastScan(
        Index* quantizer,
        AdditiveQuantizer* aq,
        size_t d,
        size_t nlist,
        MetricType metric,
        int bbs)
        : IndexIVFAQFastScan(quantizer, aq, d, nlist, metric, bbs) {}

IndexIVFAQFastScan::IndexIVFAQFastScan() {
    bbs = 0;
    M2 = 0;
    aq = nullptr;
    aq_norm = nullptr;

    is_trained = false;
}

IndexIVFAQFastScan::~IndexIVFAQFastScan() {}

/*********************************************************
 * Training
 *********************************************************/

void IndexIVFAQFastScan::train_residual(idx_t n, const float* x_in) {
    if (aq->is_trained) {
        return;
    }

    const int seed = 0x12345;
    size_t nt = n;
    const float* x = fvecs_maybe_subsample(
            d, &nt, max_training_points, x_in, verbose, seed);
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
    AlignedTable<float> residuals;
    std::vector<idx_t> assign(n);

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
    if (metric_type == METRIC_L2) {
        std::vector<float> decoded_x(n * d);
        std::vector<uint8_t> x_codes(n * aq->code_size);
        aq->compute_codes(trainset, x_codes.data(), n);
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
        estimate_norm_scale(n, x);
    }
}

void IndexIVFAQFastScan::estimate_norm_scale(idx_t n, const float* x_in) {
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

    std::unique_ptr<idx_t[]> coarse_ids(new idx_t[n]);
    std::unique_ptr<float[]> coarse_dis(new float[n]);
    quantizer->search(n, x, 1, coarse_dis.get(), coarse_ids.get());

    size_t dim12 = ksub * M;
    AlignedTable<float> dis_tables;
    AlignedTable<float> biases;
    compute_LUT(n, x, coarse_ids.get(), coarse_dis.get(), dis_tables, biases);

    float scale = 0;

#pragma omp parallel for reduction(+ : scale)
    for (idx_t i = 0; i < n; i++) {
        const float* lut = dis_tables.get() + i * M * ksub;
        scale += quantize_lut::aq_estimate_norm_scale(M, ksub, 2, lut);
    }
    scale /= n;

    scale = std::max(scale, 1.0f);
    scale = std::roundf(scale);
    norm_scale = (int)scale;

    if (verbose) {
        printf("estimated norm scale: %lf\n", scale);
        printf("rounded norm scale: %d\n", norm_scale);
    }
}

/*********************************************************
 * Code management functions
 *********************************************************/

void IndexIVFAQFastScan::encode_vectors(
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

    std::vector<float> residuals;
    residuals.resize(n * d);

#pragma omp parallel for if (n > 1000)
    for (size_t i = 0; i < n; i++) {
        if (list_nos[i] < 0) {
            memset(residuals.data() + i * d, 0, sizeof(residuals[0]) * d);
        } else {
            quantizer->compute_residual(
                    x + i * d, residuals.data() + i * d, list_nos[i]);
        }
    }

    if (metric_type == METRIC_INNER_PRODUCT) {
        aq->compute_codes(residuals.data(), codes, n);
    } else {
        // compute coarse centroids
        std::vector<float> centroids(n * d);

#pragma omp parallel for if (n > 1000)
        for (idx_t i = 0; i < n; i++) {
            auto c = centroids.data() +
                    i * d quantizer->reconstruct(list_nos[i], c);
        }

        aq->compute_codes(residuals.data(), codes, n, centroids.data());
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

void IndexIVFAQFastScan::add_with_ids(
        idx_t n,
        const float* x,
        const idx_t* xids) {
    FAISS_THROW_IF_NOT(is_trained);
    aq->verbose = verbose;

    // do some blocking to avoid excessive allocs
    constexpr idx_t bs = 65536;
    if (n > bs) {
        bool verbose = this->verbose;
        double t0 = getmillisecs();
        for (idx_t i0 = 0; i0 < n; i0 += bs) {
            idx_t i1 = std::min(n, i0 + bs);
            if (verbose) {
                double t1 = getmillisecs();
                double elapsed_time = (t1 - t0) / 1000;
                double total_time = 0;
                if (i0 != 0) {
                    total_time = elapsed_time / i0 * n;
                }
                size_t mem = get_mem_usage_kb() / (1 << 10);

                printf("\rIndexIVFAQFastScan::add_with_ids %zd/%zd, time %.2f/%.2f, RSS %zdMB",
                       size_t(i1),
                       size_t(n),
                       elapsed_time,
                       total_time,
                       mem);
                fflush(stdout);
                if (i1 == n || i0 == 0) {
                    printf("\n");
                }
                this->verbose = (i0 == 0);
            }
            add_with_ids(i1 - i0, x + i0 * d, xids ? xids + i0 : nullptr);
            this->verbose = verbose;
        }
        return;
    }
    InterruptCallback::check();

    AlignedTable<uint8_t> codes(n * code_size);
    direct_map.check_can_add(xids);
    std::unique_ptr<idx_t[]> idx(new idx_t[n]);
    quantizer->assign(n, x, idx.get());
    size_t nadd = 0, nminus1 = 0;

    for (size_t i = 0; i < n; i++) {
        if (idx[i] < 0)
            nminus1++;
    }

    AlignedTable<uint8_t> flat_codes(n * code_size);
    encode_vectors(n, x, idx.get(), flat_codes.get());

    DirectMapAdd dm_adder(direct_map, n, xids);
    BlockInvertedLists* bil = dynamic_cast<BlockInvertedLists*>(invlists);
    FAISS_THROW_IF_NOT_MSG(bil, "only block inverted lists supported");

    // prepare batches
    std::vector<idx_t> order(n);
    for (idx_t i = 0; i < n; i++) {
        order[i] = i;
    }

    // TODO should not need stable
    std::stable_sort(order.begin(), order.end(), [&idx](idx_t a, idx_t b) {
        return idx[a] < idx[b];
    });

    // TODO parallelize
    idx_t i0 = 0;
    while (i0 < n) {
        idx_t list_no = idx[order[i0]];
        idx_t i1 = i0 + 1;
        while (i1 < n && idx[order[i1]] == list_no) {
            i1++;
        }

        if (list_no == -1) {
            i0 = i1;
            continue;
        }

        // make linear array
        AlignedTable<uint8_t> list_codes((i1 - i0) * code_size);
        size_t list_size = bil->list_size(list_no);

        bil->resize(list_no, list_size + i1 - i0);

        for (idx_t i = i0; i < i1; i++) {
            size_t ofs = list_size + i - i0;
            idx_t id = xids ? xids[order[i]] : ntotal + order[i];
            dm_adder.add(order[i], list_no, ofs);
            bil->ids[list_no][ofs] = id;
            memcpy(list_codes.data() + (i - i0) * code_size,
                   flat_codes.data() + order[i] * code_size,
                   code_size);
            nadd++;

            ///// for debug only
            if (keep_orig_invlists) {
                uint8_t* code = flat_codes.data() + order[i] * code_size;
                size_t offset = orig_invlists->add_entry(list_no, id, code);
            }
        }
        pq4_pack_codes_range(
                list_codes.data(),
                M,
                list_size,
                list_size + i1 - i0,
                bbs,
                M2,
                bil->codes[list_no].data());

        i0 = i1;
    }

    ntotal += n;
}

/*********************************************************
 * search
 *********************************************************/

namespace {

template <class C, typename dis_t>
void aq_estimators_from_tables_generic(
        const IndexIVFAQFastScan& index,
        const uint8_t* codes,
        size_t ncodes,
        const dis_t* dis_table,
        const int64_t* ids,
        float bias,
        size_t k,
        int scale,
        typename C::T* heap_dis,
        int64_t* heap_ids) {
    using accu_t = typename C::T;

    for (size_t j = 0; j < ncodes; ++j) {
        BitstringReader bsr(codes + j * index.code_size, index.code_size);
        accu_t dis = bias;
        const dis_t* __restrict dt = dis_table;
        for (size_t m = 0; m < index.M - 2; m++) {
            uint64_t c = bsr.read(index.nbits);
            dis += dt[c];
            dt += index.ksub;
        }
        for (size_t m = index.M - 2; m < index.M; m++) {
            uint64_t c = bsr.read(index.nbits);
            dis += dt[c] * scale;
            dt += index.ksub;
        }

        if (C::cmp(heap_dis[0], dis)) {
            heap_pop<C>(k, heap_dis, heap_ids);
            heap_push<C>(k, heap_dis, heap_ids, dis, ids[j]);
        }
    }
}

using idx_t = Index::idx_t;
using namespace quantize_lut;

} // anonymous namespace

/*********************************************************
 * Look-Up Table functions
 *********************************************************/

/**
 * d(x, y) = || x - (y_c + \sum_i y_i) ||^2
 *         = || x ||^2 - 2 <x, y_c> - 2 \sum_i<x, y_i> + || y ||^2
 */

void IndexIVFAQFastScan::compute_LUT(
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

    // coef * <x, y_c>
    biases.resize(n * nprobe);

#pragma omp parallel
    {
        std::vector<float> centroid(d);
        float* c = centroid.data();

#pragma omp for
        for (size_t ij = 0; ij < n * nprobe; ij++) {
            int i = ij / nprobe;
            quantizer->reconstruct(coarse_ids[ij], c);
            biases[ij] = coef * fvec_inner_product(c, x + i * d, d);
        }
    }

    if (metric_type == METRIC_L2) {
        ////// TODO: set ld in compute_LUT

        const size_t norm_dim12 = 2 * ksub;

        // inner product look-up tables
        aq->compute_LUT(n, x, dis_tables.data(), -2.0f, dim12);

        // norm look-up tables
        const float* norm_lut = aq->norm_tabs.data();
        FAISS_THROW_IF_NOT(aaq->norm_tabs.size() == norm_dim12);

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

void IndexIVFAQFastScan::compute_LUT_uint8(
        size_t n,
        const float* x,
        const idx_t* coarse_ids,
        const float* coarse_dis,
        AlignedTable<uint8_t>& dis_tables,
        AlignedTable<uint16_t>& biases,
        float* normalizers) const {
    const IndexIVFAQFastScan& ivfpq = *this;
    AlignedTable<float> dis_tables_float;
    AlignedTable<float> biases_float;

    uint64_t t0 = get_cy();
    compute_LUT(n, x, coarse_ids, coarse_dis, dis_tables_float, biases_float);
    IVFAQFastScan_stats.t_compute_distance_tables += get_cy() - t0;

    constexpr bool lut_is_3d = false;
    size_t dim12 = ksub * M;
    size_t dim12_2 = ksub * M2;
    size_t M_norm = aq_norm ? aq_norm->M : 0;

    dis_tables.resize(n * dim12_2);
    if (biases_float.get()) {
        biases.resize(n * nprobe);
    }
    uint64_t t1 = get_cy();

#pragma omp parallel for if (n > 100)
    for (int64_t i = 0; i < n; i++) {
        float* t_in = dis_tables_float.get() + i * dim12;
        const float* b_in = nullptr;
        uint8_t* t_out = dis_tables.get() + i * dim12_2;
        uint16_t* b_out = nullptr;
        if (biases_float.get()) {
            b_in = biases_float.get() + i * nprobe;
            b_out = biases.get() + i * nprobe;
        }

        // rescale the norm tables
        if (rescale_norm && norm_scale > 1) {
            for (size_t m = M - M_norm; m < M; m++) {
                float* tab = t_in + m * ksub;
                for (int j = 0; j < ksub; j++) {
                    tab[j] /= norm_scale;
                }
            }
        }

        aq_quantize_LUT_and_bias(
                nprobe,
                M,
                ksub,
                t_in,
                b_in,
                rescale_norm ? M_norm : 0,
                norm_scale,
                t_out,
                M2,
                b_out,
                normalizers + 2 * i,
                normalizers + 2 * i + 1);
    }
    IVFAQFastScan_stats.t_round += get_cy() - t1;
}

/*********************************************************
 * Search functions
 *********************************************************/

template <bool is_max>
void IndexIVFAQFastScan::search_dispatch_implem(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    using Cfloat = typename std::conditional<
            is_max,
            CMax<float, int64_t>,
            CMin<float, int64_t>>::type;

    using C = typename std::conditional<
            is_max,
            CMax<uint16_t, int64_t>,
            CMin<uint16_t, int64_t>>::type;

    if (n == 0) {
        return;
    }

    // actual implementation used
    int impl = implem;

    if (impl == 0) {
        if (bbs == 32) {
            impl = 12;
        } else {
            impl = 10;
        }
        if (k > 20) {
            impl++;
        }
    }

    if (impl == 1) {
        search_implem_1<Cfloat>(n, x, k, distances, labels);
    } else if (impl == 2) {
        search_implem_2<C>(n, x, k, distances, labels);

    } else if (impl >= 10 && impl <= 13) {
        size_t ndis = 0, nlist_visited = 0;

        if (n < 2) {
            if (impl == 12 || impl == 13) {
                search_implem_12<C>(
                        n,
                        x,
                        k,
                        distances,
                        labels,
                        impl,
                        &ndis,
                        &nlist_visited);
            } else {
                search_implem_10<C>(
                        n,
                        x,
                        k,
                        distances,
                        labels,
                        impl,
                        &ndis,
                        &nlist_visited);
            }
        } else {
            // explicitly slice over threads
            int nslice;
            if (n <= omp_get_max_threads()) {
                nslice = n;
            } else {
                // LUTs unlikely to be a limiting factor
                nslice = omp_get_max_threads();
            }

#pragma omp parallel for reduction(+ : ndis, nlist_visited)
            for (int slice = 0; slice < nslice; slice++) {
                idx_t i0 = n * slice / nslice;
                idx_t i1 = n * (slice + 1) / nslice;
                float* dis_i = distances + i0 * k;
                idx_t* lab_i = labels + i0 * k;
                if (impl == 12 || impl == 13) {
                    search_implem_12<C>(
                            i1 - i0,
                            x + i0 * d,
                            k,
                            dis_i,
                            lab_i,
                            impl,
                            &ndis,
                            &nlist_visited);
                } else {
                    search_implem_10<C>(
                            i1 - i0,
                            x + i0 * d,
                            k,
                            dis_i,
                            lab_i,
                            impl,
                            &ndis,
                            &nlist_visited);
                }
            }
        }
        indexIVF_stats.nq += n;
        indexIVF_stats.ndis += ndis;
        indexIVF_stats.nlist += nlist_visited;
    } else {
        FAISS_THROW_FMT("implem %d does not exist", implem);
    }
}

void IndexIVFAQFastScan::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    FAISS_THROW_IF_NOT(k > 0);

    if (metric_type == METRIC_L2) {
        search_dispatch_implem<true>(n, x, k, distances, labels);
    } else {
        search_dispatch_implem<false>(n, x, k, distances, labels);
    }
}

template <class C>
void IndexIVFAQFastScan::search_implem_1(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    FAISS_THROW_IF_NOT(orig_invlists);

    std::unique_ptr<idx_t[]> coarse_ids(new idx_t[n * nprobe]);
    std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

    quantizer->search(n, x, nprobe, coarse_dis.get(), coarse_ids.get());

    size_t dim12 = ksub * M;
    AlignedTable<float> dis_tables;
    AlignedTable<float> biases;

    compute_LUT(n, x, coarse_ids.get(), coarse_dis.get(), dis_tables, biases);

    size_t ndis = 0, nlist_visited = 0;

#pragma omp parallel for reduction(+ : ndis, nlist_visited)
    for (idx_t i = 0; i < n; i++) {
        int64_t* heap_ids = labels + i * k;
        float* heap_dis = distances + i * k;
        heap_heapify<C>(k, heap_dis, heap_ids);

        float* lut = dis_tables.get() + i * dim12;
        for (idx_t j = 0; j < nprobe; j++) {
            idx_t list_no = coarse_ids[i * nprobe + j];
            if (list_no < 0)
                continue;
            size_t list_size = orig_invlists->list_size(list_no);
            if (list_size == 0)
                continue;
            InvertedLists::ScopedCodes codes(orig_invlists, list_no);
            InvertedLists::ScopedIds ids(orig_invlists, list_no);

            float bias = biases.get() ? biases[i * nprobe + j] : 0;

            aq_estimators_from_tables_generic<C>(
                    *this,
                    codes.get(),
                    list_size,
                    lut,
                    ids.get(),
                    bias,
                    k,
                    1,
                    heap_dis,
                    heap_ids);

            nlist_visited++;
            ndis++;
        }
        heap_reorder<C>(k, heap_dis, heap_ids);
    }
    indexIVF_stats.nq += n;
    indexIVF_stats.ndis += ndis;
    indexIVF_stats.nlist += nlist_visited;
}

template <class C>
void IndexIVFAQFastScan::search_implem_2(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    FAISS_THROW_IF_NOT(orig_invlists);

    std::unique_ptr<idx_t[]> coarse_ids(new idx_t[n * nprobe]);
    std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

    quantizer->search(n, x, nprobe, coarse_dis.get(), coarse_ids.get());

    size_t dim12 = ksub * M2;
    AlignedTable<uint8_t> dis_tables;
    AlignedTable<uint16_t> biases;
    std::unique_ptr<float[]> normalizers(new float[2 * n]);

    compute_LUT_uint8(
            n,
            x,
            coarse_ids.get(),
            coarse_dis.get(),
            dis_tables,
            biases,
            normalizers.get());

    size_t ndis = 0, nlist_visited = 0;

#pragma omp parallel for reduction(+ : ndis, nlist_visited)
    for (idx_t i = 0; i < n; i++) {
        std::vector<uint16_t> tmp_dis(k);
        int64_t* heap_ids = labels + i * k;
        uint16_t* heap_dis = tmp_dis.data();
        heap_heapify<C>(k, heap_dis, heap_ids);

        const uint8_t* lut = dis_tables.get() + i * dim12;

        for (idx_t j = 0; j < nprobe; j++) {
            idx_t list_no = coarse_ids[i * nprobe + j];
            if (list_no < 0)
                continue;
            size_t list_size = orig_invlists->list_size(list_no);
            if (list_size == 0)
                continue;
            InvertedLists::ScopedCodes codes(orig_invlists, list_no);
            InvertedLists::ScopedIds ids(orig_invlists, list_no);

            uint16_t bias = biases.get() ? biases[i * nprobe + j] : 0;

            aq_estimators_from_tables_generic<C>(
                    *this,
                    codes.get(),
                    list_size,
                    lut,
                    ids.get(),
                    bias,
                    k,
                    rescale_norm ? norm_scale : 1,
                    heap_dis,
                    heap_ids);

            nlist_visited++;
            ndis += list_size;
        }
        heap_reorder<C>(k, heap_dis, heap_ids);
        // convert distances to float
        {
            float one_a = 1 / normalizers[2 * i], b = normalizers[2 * i + 1];
            if (skip & 16) {
                one_a = 1;
                b = 0;
            }
            float* heap_dis_float = distances + i * k;
            for (int j = 0; j < k; j++) {
                heap_dis_float[j] = b + heap_dis[j] * one_a;
            }
        }
    }
    indexIVF_stats.nq += n;
    indexIVF_stats.ndis += ndis;
    indexIVF_stats.nlist += nlist_visited;
}

template <class C>
void IndexIVFAQFastScan::search_implem_10(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int impl,
        size_t* ndis_out,
        size_t* nlist_out) const {
    memset(distances, -1, sizeof(float) * k * n);
    memset(labels, -1, sizeof(idx_t) * k * n);
    size_t M_norm = aq_norm ? aq_norm->M : 0;

    using HeapHC = HeapHandler<C, true>;
    using ReservoirHC = ReservoirHandler<C, true>;
    using SingleResultHC = SingleResultHandler<C, true>;

    std::unique_ptr<idx_t[]> coarse_ids(new idx_t[n * nprobe]);
    std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

    uint64_t times[10];
    memset(times, 0, sizeof(times));
    int ti = 0;
#define TIC times[ti++] = get_cy()
    TIC;

    quantizer->search(n, x, nprobe, coarse_dis.get(), coarse_ids.get());

    TIC;

    size_t dim12 = ksub * M2;
    AlignedTable<uint8_t> dis_tables;
    AlignedTable<uint16_t> biases;
    std::unique_ptr<float[]> normalizers(new float[2 * n]);

    compute_LUT_uint8(
            n,
            x,
            coarse_ids.get(),
            coarse_dis.get(),
            dis_tables,
            biases,
            normalizers.get());

    TIC;

    size_t ndis = 0, nlist_visited = 0;

    {
        AlignedTable<uint16_t> tmp_distances(k);
        for (idx_t i = 0; i < n; i++) {
            int qmap1[1] = {0};
            std::unique_ptr<SIMDResultHandler<C, true>> handler;

            if (k == 1) {
                handler.reset(new SingleResultHC(1, 0));
            } else if (impl == 10) {
                handler.reset(new HeapHC(
                        1, tmp_distances.get(), labels + i * k, k, 0));
            } else if (impl == 11) {
                handler.reset(new ReservoirHC(1, 0, k, 2 * k));
            } else {
                FAISS_THROW_MSG("invalid");
            }

            handler->q_map = qmap1;

            const uint8_t* LUT = dis_tables.get() + i * dim12;
            for (idx_t j = 0; j < nprobe; j++) {
                size_t ij = i * nprobe + j;
                if (biases.get()) {
                    handler->dbias = biases.get() + ij;
                }

                idx_t list_no = coarse_ids[ij];
                if (list_no < 0)
                    continue;
                size_t ls = invlists->list_size(list_no);
                if (ls == 0)
                    continue;

                InvertedLists::ScopedCodes codes(invlists, list_no);
                InvertedLists::ScopedIds ids(invlists, list_no);

                handler->ntotal = ls;
                handler->id_map = ids.get();

#define DISPATCH(classHC)                                                  \
    if (dynamic_cast<classHC*>(handler.get())) {                           \
        auto* res = static_cast<classHC*>(handler.get());                  \
        if (rescale_norm && norm_scale > 1) {                              \
            aq4_accumulate_loop(                                           \
                    1,                                                     \
                    roundup(ls, bbs),                                      \
                    bbs,                                                   \
                    M2,                                                    \
                    M_norm,                                                \
                    norm_scale,                                            \
                    codes.get(),                                           \
                    LUT,                                                   \
                    *res);                                                 \
        } else {                                                           \
            pq4_accumulate_loop(                                           \
                    1, roundup(ls, bbs), bbs, M2, codes.get(), LUT, *res); \
        }                                                                  \
    }
                DISPATCH(HeapHC)
                else DISPATCH(ReservoirHC) else DISPATCH(SingleResultHC)
#undef DISPATCH

                        nlist_visited++;
                ndis++;
            }

            handler->to_flat_arrays(
                    distances + i * k,
                    labels + i * k,
                    skip & 16 ? nullptr : normalizers.get() + i * 2);
        }
    }
    *ndis_out = ndis;
    *nlist_out = nlist;
}

template <class C>
void IndexIVFAQFastScan::search_implem_12(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int impl,
        size_t* ndis_out,
        size_t* nlist_out) const {
    if (n == 0) { // does not work well with reservoir
        return;
    }
    FAISS_THROW_IF_NOT(bbs == 32);

    std::unique_ptr<idx_t[]> coarse_ids(new idx_t[n * nprobe]);
    std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);
    size_t M_norm = aq_norm ? aq_norm->M : 0;

    uint64_t times[10];
    memset(times, 0, sizeof(times));
    int ti = 0;
#define TIC times[ti++] = get_cy()
    TIC;

    quantizer->search(n, x, nprobe, coarse_dis.get(), coarse_ids.get());

    TIC;

    size_t dim12 = ksub * M2;
    AlignedTable<uint8_t> dis_tables;
    AlignedTable<uint16_t> biases;
    std::unique_ptr<float[]> normalizers(new float[2 * n]);

    compute_LUT_uint8(
            n,
            x,
            coarse_ids.get(),
            coarse_dis.get(),
            dis_tables,
            biases,
            normalizers.get());

    TIC;

    struct QC {
        int qno;     // sequence number of the query
        int list_no; // list to visit
        int rank;    // this is the rank'th result of the coarse quantizer
    };

    std::vector<QC> qcs;
    {
        int ij = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nprobe; j++) {
                if (coarse_ids[ij] >= 0) {
                    qcs.push_back(QC{i, int(coarse_ids[ij]), int(j)});
                }
                ij++;
            }
        }
        std::sort(qcs.begin(), qcs.end(), [](const QC& a, const QC& b) {
            return a.list_no < b.list_no;
        });
    }
    TIC;

    // prepare the result handlers

    std::unique_ptr<SIMDResultHandler<C, true>> handler;
    AlignedTable<uint16_t> tmp_distances;

    using HeapHC = HeapHandler<C, true>;
    using ReservoirHC = ReservoirHandler<C, true>;
    using SingleResultHC = SingleResultHandler<C, true>;

    if (k == 1) {
        handler.reset(new SingleResultHC(n, 0));
    } else if (impl == 12) {
        tmp_distances.resize(n * k);
        handler.reset(new HeapHC(n, tmp_distances.get(), labels, k, 0));
    } else if (impl == 13) {
        handler.reset(new ReservoirHC(n, 0, k, 2 * k));
    }

    int qbs2 = this->qbs2 ? this->qbs2 : 11;

    std::vector<uint16_t> tmp_bias;
    if (biases.get()) {
        tmp_bias.resize(qbs2);
        handler->dbias = tmp_bias.data();
    }
    TIC;

    size_t ndis = 0;

    size_t i0 = 0;
    uint64_t t_copy_pack = 0, t_scan = 0;
    while (i0 < qcs.size()) {
        uint64_t tt0 = get_cy();

        // find all queries that access this inverted list
        int list_no = qcs[i0].list_no;
        size_t i1 = i0 + 1;

        while (i1 < qcs.size() && i1 < i0 + qbs2) {
            if (qcs[i1].list_no != list_no) {
                break;
            }
            i1++;
        }

        size_t list_size = invlists->list_size(list_no);

        if (list_size == 0) {
            i0 = i1;
            continue;
        }

        // re-organize LUTs and biases into the right order
        int nc = i1 - i0;

        std::vector<int> q_map(nc), lut_entries(nc);
        AlignedTable<uint8_t> LUT(nc * dim12);
        memset(LUT.get(), -1, nc * dim12);
        int qbs = pq4_preferred_qbs(nc);

        for (size_t i = i0; i < i1; i++) {
            const QC& qc = qcs[i];
            q_map[i - i0] = qc.qno;
            int ij = qc.qno * nprobe + qc.rank;
            lut_entries[i - i0] = qc.qno;
            if (biases.get()) {
                tmp_bias[i - i0] = biases[ij];
            }
        }
        pq4_pack_LUT_qbs_q_map(
                qbs, M2, dis_tables.get(), lut_entries.data(), LUT.get());

        // access the inverted list

        ndis += (i1 - i0) * list_size;

        InvertedLists::ScopedCodes codes(invlists, list_no);
        InvertedLists::ScopedIds ids(invlists, list_no);

        // prepare the handler

        handler->ntotal = list_size;
        handler->q_map = q_map.data();
        handler->id_map = ids.get();
        uint64_t tt1 = get_cy();

#define DISPATCH(classHC)                                              \
    if (dynamic_cast<classHC*>(handler.get())) {                       \
        auto* res = static_cast<classHC*>(handler.get());              \
        if (rescale_norm && norm_scale > 1) {                          \
            aq4_accumulate_loop_qbs(                                   \
                    qbs,                                               \
                    list_size,                                         \
                    M2,                                                \
                    M_norm,                                            \
                    norm_scale,                                        \
                    codes.get(),                                       \
                    LUT.get(),                                         \
                    *res);                                             \
        } else {                                                       \
            pq4_accumulate_loop_qbs(                                   \
                    qbs, list_size, M2, codes.get(), LUT.get(), *res); \
        }                                                              \
    }
        DISPATCH(HeapHC)
        else DISPATCH(ReservoirHC) else DISPATCH(SingleResultHC)

                // prepare for next loop
                i0 = i1;

        uint64_t tt2 = get_cy();
        t_copy_pack += tt1 - tt0;
        t_scan += tt2 - tt1;
    }
    TIC;

    // labels is in-place for HeapHC
    handler->to_flat_arrays(
            distances, labels, skip & 16 ? nullptr : normalizers.get());

    TIC;

    // these stats are not thread-safe

    for (int i = 1; i < ti; i++) {
        IVFAQFastScan_stats.times[i] += times[i] - times[i - 1];
    }
    IVFAQFastScan_stats.t_copy_pack += t_copy_pack;
    IVFAQFastScan_stats.t_scan += t_scan;

    if (auto* rh = dynamic_cast<ReservoirHC*>(handler.get())) {
        for (int i = 0; i < 4; i++) {
            IVFAQFastScan_stats.reservoir_times[i] += rh->times[i];
        }
    }

    *ndis_out = ndis;
    *nlist_out = nlist;
}

void IndexIVFAQFastScan::reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    // unpack codes
    InvertedLists::ScopedCodes list_codes(invlists, list_no);
    std::vector<uint8_t> code(code_size, 0);
    BitstringWriter bsw(code.data(), code_size);
    for (size_t m = 0; m < M; m++) {
        uint8_t c =
                pq4_get_packed_element(list_codes.get(), bbs, M2, offset, m);
        bsw.write(c, nbits);
    }
    aq->decode(code.data(), recons, 1);

    // add centroid to it
    std::vector<float> centroid(d);
    quantizer->reconstruct(list_no, centroid.data());
    for (int i = 0; i < d; ++i) {
        recons[i] += centroid[i];
    }
}

void IndexIVFAQFastScan::reconstruct_orig_invlists() {
    FAISS_THROW_IF_NOT(orig_invlists != nullptr);
    FAISS_THROW_IF_NOT(orig_invlists->list_size(0) == 0);

    for (size_t list_no = 0; list_no < nlist; list_no++) {
        InvertedLists::ScopedCodes codes(invlists, list_no);
        InvertedLists::ScopedIds ids(invlists, list_no);
        size_t list_size = orig_invlists->list_size(list_no);
        std::vector<uint8_t> code(code_size, 0);

        for (size_t offset = 0; offset < list_size; offset++) {
            // unpack codes
            BitstringWriter bsw(code.data(), code_size);
            for (size_t m = 0; m < M; m++) {
                uint8_t c =
                        pq4_get_packed_element(codes.get(), bbs, M2, offset, m);
                bsw.write(c, nbits);
            }

            // get id
            idx_t id = ids.get()[offset];

            orig_invlists->add_entry(list_no, id, code.data());
        }
    }
}

IndexIVFLSQFastScan::IndexIVFLSQFastScan(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t M,
        size_t M_norm,
        MetricType metric,
        int bbs)
        : lsq(d, M, 4),
          norm_lsq(d, M_norm, 4),
          IndexIVFAQFastScan(
                  quantizer,
                  &lsq,
                  &norm_lsq,
                  d,
                  nlist,
                  metric,
                  bbs) {}

IndexIVFLSQFastScan::IndexIVFLSQFastScan() {}

IndexIVFLSQFastScan::~IndexIVFLSQFastScan() {}

IVFAQFastScanStats IVFAQFastScan_stats;

} // namespace faiss