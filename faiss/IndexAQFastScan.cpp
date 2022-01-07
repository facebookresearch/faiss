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
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

#include <faiss/impl/pq4_fast_scan.h>
#include <faiss/impl/simd_result_handlers.h>
#include <faiss/utils/quantize_lut.h>

namespace faiss {

using namespace simd_result_handlers;

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
    this->aq = aq;
    this->bbs = bbs;
    d = aq->d;
    metric_type = metric;
    nbits = aq->nbits[0];
    ksub = (1 << nbits);
    ntotal2 = 0;

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

    if (metric_type == METRIC_L2) {
        M = aq->M + 2; // 2x4 bits AQ
    } else {
        M = aq->M;
    }

    is_trained = false;
    M2 = roundup(M, 2);
    code_size = (M * nbits + 7) / 8;

    max_training_points = 1024 * ksub * M;
}

IndexAQFastScan::IndexAQFastScan() : bbs(0), ntotal2(0), M2(0) {
    is_trained = false;
    aq = nullptr;
}

IndexAQFastScan::IndexAQFastScan(const IndexAdditiveQuantizer& orig, int bbs)
        : Index(orig.aq->d, orig.metric_type) {
    aq = orig.aq;
    FAISS_THROW_IF_NOT(aq->nbits[0] == 4);
    FAISS_THROW_IF_NOT_MSG(
            orig.metric_type == METRIC_INNER_PRODUCT ||
                    aq->search_type == AdditiveQuantizer::ST_norm_lsq2x4 ||
                    aq->search_type == AdditiveQuantizer::ST_norm_rq2x4,
            "Search type must be lsq2x4 or rq2x4");

    ntotal = orig.ntotal;
    is_trained = orig.is_trained;
    orig_codes = orig.codes.data();

    nbits = aq->nbits[0];
    ksub = (1 << nbits);
    this->bbs = bbs;

    qbs = 0; // means use default

    // pack the codes
    M = aq->M + 2;
    max_training_points = 1024 * ksub * M;

    FAISS_THROW_IF_NOT(bbs % 32 == 0);
    M2 = roundup(M, 2);
    code_size = (M * nbits + 7) / 8;
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
            d, &nt, max_training_points, x_in, verbose, seed);
    n = nt;
    if (verbose) {
        printf("training additive quantizer on %zd vectors\n", nt);
    }

    aq->verbose = verbose;
    aq->train(n, x);

    is_trained = true;
}

void IndexAQFastScan::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    if (implem == 0x22 || implem == 2) {
        FAISS_THROW_IF_NOT(orig_codes != nullptr);
    }

    aq->verbose = verbose;

    // do some blocking to avoid excessive allocs
    constexpr idx_t bs = 65536;
    if (n > bs) {
        bool verbose = this->verbose;
        for (idx_t i0 = 0; i0 < n; i0 += bs) {
            idx_t i1 = std::min(n, i0 + bs);
            if (verbose) {
                printf("\rIndexAQFastScan::add_with_ids %zd/%zd",
                       size_t(i1),
                       size_t(n));
                fflush(stdout);
                if (i1 == n || i0 == 0) {
                    printf("\n");
                }
                this->verbose = (i0 == 0);
            }
            add(i1 - i0, x + i0 * d);
        }
        this->verbose = verbose;
        return;
    }
    InterruptCallback::check();

    AlignedTable<uint8_t> tmp_codes(n * code_size);
    compute_codes(tmp_codes.get(), n, x);

    ntotal2 = roundup(ntotal + n, bbs);
    size_t new_size = ntotal2 * M2 / 2; // assume nbits = 4
    size_t old_size = codes.size();
    if (new_size > old_size) {
        codes.resize(new_size);
        memset(codes.get() + old_size, 0, new_size - old_size);
    }

    pq4_pack_codes_range(
            tmp_codes.get(), M, ntotal, ntotal + n, bbs, M2, codes.get());

    ntotal += n;
}

void IndexAQFastScan::compute_codes(uint8_t* tmp_codes, idx_t n, const float* x)
        const {
    aq->compute_codes(x, tmp_codes, n);
}

void IndexAQFastScan::reset() {
    codes.resize(0);
    ntotal = 0;
}

namespace {

template <class C, typename dis_t>
void aq_estimators_from_tables_generic(
        const IndexAQFastScan& index,
        const uint8_t* codes,
        size_t ncodes,
        const dis_t* dis_table,
        size_t k,
        typename C::T* heap_dis,
        int64_t* heap_ids) {
    using accu_t = typename C::T;

    for (size_t j = 0; j < ncodes; ++j) {
        BitstringReader bsr(codes + j * index.code_size, index.code_size);
        accu_t dis = 0;
        const dis_t* __restrict dt = dis_table;
        for (size_t m = 0; m < index.M; m++) {
            uint64_t c = bsr.read(index.nbits);
            dis += dt[c];
            dt += index.ksub;
        }

        if (C::cmp(heap_dis[0], dis)) {
            heap_pop<C>(k, heap_dis, heap_ids);
            heap_push<C>(k, heap_dis, heap_ids, dis, j);
        }
    }
}

template <class VectorDistance, class ResultHandler>
void search_with_decompress(
        const IndexAQFastScan& index,
        size_t ntotal,
        const float* xq,
        VectorDistance& vd,
        ResultHandler& res) {
    using SingleResultHandler = typename ResultHandler::SingleResultHandler;
    const uint8_t* codes = index.orig_codes;

#pragma omp parallel for
    for (int64_t q = 0; q < res.nq; q++) {
        SingleResultHandler resi(res);
        resi.begin(q);
        std::vector<float> tmp(index.d);
        const float* x = xq + index.d * q;
        for (size_t i = 0; i < ntotal; i++) {
            index.aq->decode(codes + i * index.code_size, tmp.data(), 1);
            float dis = vd(x, tmp.data());
            resi.add_result(dis, i);
        }
        resi.end();
    }
}

} // anonymous namespace

using namespace quantize_lut;

void IndexAQFastScan::compute_quantized_LUT(
        idx_t n,
        const float* x,
        uint8_t* lut,
        float* normalizers) const {
    size_t dim12 = ksub * M;
    std::unique_ptr<float[]> dis_tables(new float[n * dim12]);
    compute_LUT(dis_tables.get(), n, x);

    for (uint64_t i = 0; i < n; i++) {
        round_uint8_per_column(
                dis_tables.get() + i * dim12,
                M,
                ksub,
                &normalizers[2 * i],
                &normalizers[2 * i + 1]);
    }

    for (uint64_t i = 0; i < n; i++) {
        const float* t_in = dis_tables.get() + i * dim12;
        uint8_t* t_out = lut + i * M2 * ksub;

        for (int j = 0; j < dim12; j++) {
            t_out[j] = int(t_in[j]);
        }
        memset(t_out + dim12, 0, (M2 - M) * ksub);
    }
}

/******************************************************************************
 * Search driver routine
 ******************************************************************************/

void IndexAQFastScan::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    FAISS_THROW_IF_NOT(k > 0);

    if (implem == 0x22) {
        if (metric_type == METRIC_L2) {
            using VD = VectorDistance<METRIC_L2>;
            VD vd = {size_t(d), metric_arg};
            HeapResultHandler<VD::C> rh(n, distances, labels, k);
            search_with_decompress(*this, ntotal, x, vd, rh);
        } else {
            using VD = VectorDistance<METRIC_INNER_PRODUCT>;
            VD vd = {size_t(d), metric_arg};
            HeapResultHandler<VD::C> rh(n, distances, labels, k);
            search_with_decompress(*this, ntotal, x, vd, rh);
        }
        return;
    }

    if (metric_type == METRIC_L2) {
        search_dispatch_implem<true>(n, x, k, distances, labels);
    } else {
        search_dispatch_implem<false>(n, x, k, distances, labels);
    }
}

template <bool is_max>
void IndexAQFastScan::search_dispatch_implem(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    using Cfloat = typename std::conditional<
            is_max,
            CMax<float, int64_t>,
            CMin<float, int64_t>>::type;

    using C = typename std::
            conditional<is_max, CMax<uint16_t, int>, CMin<uint16_t, int>>::type;

    if (n == 0) {
        return;
    }

    // actual implementation used
    int impl = implem;

    if (impl == 0) {
        if (bbs == 32) {
            impl = 12;
        } else {
            impl = 14;
        }
        if (k > 20) {
            impl++;
        }
    }

    if (implem == 1) {
        FAISS_THROW_MSG("Not implemented yet.");
        // FAISS_THROW_IF_NOT(orig_codes);
        // FAISS_THROW_IF_NOT(is_max);
        // float_maxheap_array_t res = {size_t(n), size_t(k), labels,
        // distances}; aq->search(x, n, orig_codes, ntotal, &res, true);
    } else if (implem == 2 || implem == 3 || implem == 4) {
        FAISS_THROW_IF_NOT(orig_codes != nullptr);

        const size_t dim12 = ksub * M;
        std::unique_ptr<float[]> dis_tables(new float[n * dim12]);
        compute_LUT(dis_tables.get(), n, x);

        std::vector<float> normalizers(n * 2);

        if (implem == 2) {
            // default float
        } else if (implem == 3 || implem == 4) {
            for (uint64_t i = 0; i < n; i++) {
                round_uint8_per_column(
                        dis_tables.get() + i * dim12,
                        M,
                        ksub,
                        &normalizers[2 * i],
                        &normalizers[2 * i + 1]);
            }
        }

#pragma omp parallel for if (n > 1000)
        for (int64_t i = 0; i < n; i++) {
            int64_t* heap_ids = labels + i * k;
            float* heap_dis = distances + i * k;

            heap_heapify<Cfloat>(k, heap_dis, heap_ids);

            aq_estimators_from_tables_generic<Cfloat>(
                    *this,
                    orig_codes,
                    ntotal,
                    dis_tables.get() + i * dim12,
                    k,
                    heap_dis,
                    heap_ids);

            heap_reorder<Cfloat>(k, heap_dis, heap_ids);

            if (implem == 4) {
                float a = normalizers[2 * i];
                float b = normalizers[2 * i + 1];

                for (int j = 0; j < k; j++) {
                    heap_dis[j] = heap_dis[j] / a + b;
                }
            }
        }
    } else if (impl >= 12 && impl <= 15) {
        FAISS_THROW_IF_NOT(ntotal < INT_MAX);
        int nt = std::min(omp_get_max_threads(), int(n));
        if (nt < 2) {
            if (impl == 12 || impl == 13) {
                search_implem_12<C>(n, x, k, distances, labels, impl);
            } else {
                search_implem_14<C>(n, x, k, distances, labels, impl);
            }
        } else {
            // explicitly slice over threads
#pragma omp parallel for num_threads(nt)
            for (int slice = 0; slice < nt; slice++) {
                idx_t i0 = n * slice / nt;
                idx_t i1 = n * (slice + 1) / nt;
                float* dis_i = distances + i0 * k;
                idx_t* lab_i = labels + i0 * k;
                if (impl == 12 || impl == 13) {
                    search_implem_12<C>(
                            i1 - i0, x + i0 * d, k, dis_i, lab_i, impl);
                } else {
                    search_implem_14<C>(
                            i1 - i0, x + i0 * d, k, dis_i, lab_i, impl);
                }
            }
        }
    } else {
        FAISS_THROW_FMT("invalid implem %d impl=%d", implem, impl);
    }
}

void IndexAQFastScan::compute_LUT(float* lut, idx_t n, const float* x) const {
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

template <class C>
void IndexAQFastScan::search_implem_12(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int impl) const {
    FAISS_THROW_IF_NOT(bbs == 32);

    // handle qbs2 blocking by recursive call
    int64_t qbs2 = this->qbs == 0 ? 11 : pq4_qbs_to_nq(this->qbs);
    if (n > qbs2) {
        for (int64_t i0 = 0; i0 < n; i0 += qbs2) {
            int64_t i1 = std::min(i0 + qbs2, n);
            search_implem_12<C>(
                    i1 - i0,
                    x + d * i0,
                    k,
                    distances + i0 * k,
                    labels + i0 * k,
                    impl);
        }
        return;
    }

    size_t dim12 = ksub * M2;
    AlignedTable<uint8_t> quantized_dis_tables(n * dim12);
    std::unique_ptr<float[]> normalizers(new float[2 * n]);

    if (skip & 1) {
        quantized_dis_tables.clear();
    } else {
        compute_quantized_LUT(
                n, x, quantized_dis_tables.get(), normalizers.get());
    }

    AlignedTable<uint8_t> LUT(n * dim12);

    // block sizes are encoded in qbs, 4 bits at a time

    // caution: we override an object field
    int qbs = this->qbs;

    if (n != pq4_qbs_to_nq(qbs)) {
        qbs = pq4_preferred_qbs(n);
    }

    int LUT_nq =
            pq4_pack_LUT_qbs(qbs, M2, quantized_dis_tables.get(), LUT.get());
    FAISS_THROW_IF_NOT(LUT_nq == n);

    if (k == 1) {
        SingleResultHandler<C> handler(n, ntotal);
        if (skip & 4) {
            // pass
        } else {
            handler.disable = bool(skip & 2);
            pq4_accumulate_loop_qbs(
                    qbs, ntotal2, M2, codes.get(), LUT.get(), handler);
        }

        handler.to_flat_arrays(distances, labels, normalizers.get());

    } else if (impl == 12) {
        std::vector<uint16_t> tmp_dis(n * k);
        std::vector<int32_t> tmp_ids(n * k);

        if (skip & 4) {
            // skip
        } else {
            HeapHandler<C> handler(
                    n, tmp_dis.data(), tmp_ids.data(), k, ntotal);
            handler.disable = bool(skip & 2);

            pq4_accumulate_loop_qbs(
                    qbs, ntotal2, M2, codes.get(), LUT.get(), handler);

            if (!(skip & 8)) {
                handler.to_flat_arrays(distances, labels, normalizers.get());
            }
        }

    } else { // impl == 13

        ReservoirHandler<C> handler(n, ntotal, k, 2 * k);
        handler.disable = bool(skip & 2);

        if (skip & 4) {
            // skip
        } else {
            pq4_accumulate_loop_qbs(
                    qbs, ntotal2, M2, codes.get(), LUT.get(), handler);
        }

        if (!(skip & 8)) {
            handler.to_flat_arrays(distances, labels, normalizers.get());
        }

        AQFastScan_stats.t0 += handler.times[0];
        AQFastScan_stats.t1 += handler.times[1];
        AQFastScan_stats.t2 += handler.times[2];
        AQFastScan_stats.t3 += handler.times[3];
    }
}

AQFastScanStats AQFastScan_stats;

template <class C>
void IndexAQFastScan::search_implem_14(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int impl) const {
    FAISS_THROW_IF_NOT(bbs % 32 == 0);

    int qbs2 = qbs == 0 ? 4 : qbs;

    // handle qbs2 blocking by recursive call
    if (n > qbs2) {
        for (int64_t i0 = 0; i0 < n; i0 += qbs2) {
            int64_t i1 = std::min(i0 + qbs2, n);
            search_implem_14<C>(
                    i1 - i0,
                    x + d * i0,
                    k,
                    distances + i0 * k,
                    labels + i0 * k,
                    impl);
        }
        return;
    }

    size_t dim12 = ksub * M2;
    AlignedTable<uint8_t> quantized_dis_tables(n * dim12);
    std::unique_ptr<float[]> normalizers(new float[2 * n]);

    if (skip & 1) {
        quantized_dis_tables.clear();
    } else {
        compute_quantized_LUT(
                n, x, quantized_dis_tables.get(), normalizers.get());
    }

    AlignedTable<uint8_t> LUT(n * dim12);
    pq4_pack_LUT(n, M2, quantized_dis_tables.get(), LUT.get());

    if (k == 1) {
        SingleResultHandler<C> handler(n, ntotal);
        if (skip & 4) {
            // pass
        } else {
            handler.disable = bool(skip & 2);
            pq4_accumulate_loop(
                    n, ntotal2, bbs, M2, codes.get(), LUT.get(), handler);
        }
        handler.to_flat_arrays(distances, labels, normalizers.get());

    } else if (impl == 14) {
        std::vector<uint16_t> tmp_dis(n * k);
        std::vector<int32_t> tmp_ids(n * k);

        if (skip & 4) {
            // skip
        } else if (k > 1) {
            HeapHandler<C> handler(
                    n, tmp_dis.data(), tmp_ids.data(), k, ntotal);
            handler.disable = bool(skip & 2);

            pq4_accumulate_loop(
                    n, ntotal2, bbs, M2, codes.get(), LUT.get(), handler);

            if (!(skip & 8)) {
                handler.to_flat_arrays(distances, labels, normalizers.get());
            }
        }

    } else { // impl == 15

        ReservoirHandler<C> handler(n, ntotal, k, 2 * k);
        handler.disable = bool(skip & 2);

        if (skip & 4) {
            // skip
        } else {
            pq4_accumulate_loop(
                    n, ntotal2, bbs, M2, codes.get(), LUT.get(), handler);
        }

        if (!(skip & 8)) {
            handler.to_flat_arrays(distances, labels, normalizers.get());
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

IndexRQFastScan::IndexRQFastScan() : IndexRQFastScan(0, 0, 0) {}

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

IndexLSQFastScan::IndexLSQFastScan() : IndexLSQFastScan(0, 0, 0) {}

} // namespace faiss