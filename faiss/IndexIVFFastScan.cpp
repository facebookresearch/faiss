/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexIVFFastScan.h>

#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <set>

#include <omp.h>

#include <memory>

#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/LookupTableScaler.h>
#include <faiss/impl/pq4_fast_scan.h>
#include <faiss/impl/simd_result_handlers.h>
#include <faiss/invlists/BlockInvertedLists.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/quantize_lut.h>
#include <faiss/utils/utils.h>

namespace faiss {

using namespace simd_result_handlers;

inline size_t roundup(size_t a, size_t b) {
    return (a + b - 1) / b * b;
}

IndexIVFFastScan::IndexIVFFastScan(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t code_size,
        MetricType metric)
        : IndexIVF(quantizer, d, nlist, code_size, metric) {
    FAISS_THROW_IF_NOT(metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT);
}

IndexIVFFastScan::IndexIVFFastScan() {
    bbs = 0;
    M2 = 0;
    is_trained = false;
}

void IndexIVFFastScan::init_fastscan(
        size_t M,
        size_t nbits,
        size_t nlist,
        MetricType /* metric */,
        int bbs) {
    FAISS_THROW_IF_NOT(bbs % 32 == 0);
    FAISS_THROW_IF_NOT(nbits == 4);

    this->M = M;
    this->nbits = nbits;
    this->bbs = bbs;
    ksub = (1 << nbits);
    M2 = roundup(M, 2);
    code_size = M2 / 2;

    is_trained = false;
    replace_invlists(new BlockInvertedLists(nlist, bbs, bbs * M2 / 2), true);
}

IndexIVFFastScan::~IndexIVFFastScan() {}

/*********************************************************
 * Code management functions
 *********************************************************/

void IndexIVFFastScan::add_with_ids(
        idx_t n,
        const float* x,
        const idx_t* xids) {
    FAISS_THROW_IF_NOT(is_trained);

    // do some blocking to avoid excessive allocs
    constexpr idx_t bs = 65536;
    if (n > bs) {
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

                printf("IndexIVFFastScan::add_with_ids %zd/%zd, time %.2f/%.2f, RSS %zdMB\n",
                       size_t(i1),
                       size_t(n),
                       elapsed_time,
                       total_time,
                       mem);
            }
            add_with_ids(i1 - i0, x + i0 * d, xids ? xids + i0 : nullptr);
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
        if (idx[i] < 0) {
            nminus1++;
        }
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

template <class C, typename dis_t, class Scaler>
void estimators_from_tables_generic(
        const IndexIVFFastScan& index,
        const uint8_t* codes,
        size_t ncodes,
        const dis_t* dis_table,
        const int64_t* ids,
        float bias,
        size_t k,
        typename C::T* heap_dis,
        int64_t* heap_ids,
        const Scaler& scaler) {
    using accu_t = typename C::T;
    for (size_t j = 0; j < ncodes; ++j) {
        BitstringReader bsr(codes + j * index.code_size, index.code_size);
        accu_t dis = bias;
        const dis_t* __restrict dt = dis_table;
        for (size_t m = 0; m < index.M - scaler.nscale; m++) {
            uint64_t c = bsr.read(index.nbits);
            dis += dt[c];
            dt += index.ksub;
        }

        for (size_t m = 0; m < scaler.nscale; m++) {
            uint64_t c = bsr.read(index.nbits);
            dis += scaler.scale_one(dt[c]);
            dt += index.ksub;
        }

        if (C::cmp(heap_dis[0], dis)) {
            heap_pop<C>(k, heap_dis, heap_ids);
            heap_push<C>(k, heap_dis, heap_ids, dis, ids[j]);
        }
    }
}

using namespace quantize_lut;

} // anonymous namespace

/*********************************************************
 * Look-Up Table functions
 *********************************************************/

void IndexIVFFastScan::compute_LUT_uint8(
        size_t n,
        const float* x,
        const idx_t* coarse_ids,
        const float* coarse_dis,
        AlignedTable<uint8_t>& dis_tables,
        AlignedTable<uint16_t>& biases,
        float* normalizers) const {
    AlignedTable<float> dis_tables_float;
    AlignedTable<float> biases_float;

    uint64_t t0 = get_cy();
    compute_LUT(n, x, coarse_ids, coarse_dis, dis_tables_float, biases_float);
    IVFFastScan_stats.t_compute_distance_tables += get_cy() - t0;

    bool lut_is_3d = lookup_table_is_3d();
    size_t dim123 = ksub * M;
    size_t dim123_2 = ksub * M2;
    if (lut_is_3d) {
        dim123 *= nprobe;
        dim123_2 *= nprobe;
    }
    dis_tables.resize(n * dim123_2);
    if (biases_float.get()) {
        biases.resize(n * nprobe);
    }
    uint64_t t1 = get_cy();

#pragma omp parallel for if (n > 100)
    for (int64_t i = 0; i < n; i++) {
        const float* t_in = dis_tables_float.get() + i * dim123;
        const float* b_in = nullptr;
        uint8_t* t_out = dis_tables.get() + i * dim123_2;
        uint16_t* b_out = nullptr;
        if (biases_float.get()) {
            b_in = biases_float.get() + i * nprobe;
            b_out = biases.get() + i * nprobe;
        }

        quantize_LUT_and_bias(
                nprobe,
                M,
                ksub,
                lut_is_3d,
                t_in,
                b_in,
                t_out,
                M2,
                b_out,
                normalizers + 2 * i,
                normalizers + 2 * i + 1);
    }
    IVFFastScan_stats.t_round += get_cy() - t1;
}

/*********************************************************
 * Search functions
 *********************************************************/

void IndexIVFFastScan::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");
    FAISS_THROW_IF_NOT(k > 0);

    DummyScaler scaler;
    if (metric_type == METRIC_L2) {
        search_dispatch_implem<true>(n, x, k, distances, labels, scaler);
    } else {
        search_dispatch_implem<false>(n, x, k, distances, labels, scaler);
    }
}

void IndexIVFFastScan::range_search(
        idx_t,
        const float*,
        float,
        RangeSearchResult*,
        const SearchParameters*) const {
    FAISS_THROW_MSG("not implemented");
}

template <bool is_max, class Scaler>
void IndexIVFFastScan::search_dispatch_implem(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const Scaler& scaler) const {
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
        search_implem_1<Cfloat>(n, x, k, distances, labels, scaler);
    } else if (impl == 2) {
        search_implem_2<C>(n, x, k, distances, labels, scaler);

    } else if (impl >= 10 && impl <= 15) {
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
                        &nlist_visited,
                        scaler);
            } else if (impl == 14 || impl == 15) {
                search_implem_14<C>(n, x, k, distances, labels, impl, scaler);
            } else {
                search_implem_10<C>(
                        n,
                        x,
                        k,
                        distances,
                        labels,
                        impl,
                        &ndis,
                        &nlist_visited,
                        scaler);
            }
        } else {
            // explicitly slice over threads
            int nslice;
            if (n <= omp_get_max_threads()) {
                nslice = n;
            } else if (lookup_table_is_3d()) {
                // make sure we don't make too big LUT tables
                size_t lut_size_per_query =
                        M * ksub * nprobe * (sizeof(float) + sizeof(uint8_t));

                size_t max_lut_size = precomputed_table_max_bytes;
                // how many queries we can handle within mem budget
                size_t nq_ok =
                        std::max(max_lut_size / lut_size_per_query, size_t(1));
                nslice =
                        roundup(std::max(size_t(n / nq_ok), size_t(1)),
                                omp_get_max_threads());
            } else {
                // LUTs unlikely to be a limiting factor
                nslice = omp_get_max_threads();
            }
            if (impl == 14 ||
                impl == 15) { // this might require slicing if there are too
                              // many queries (for now we keep this simple)
                search_implem_14<C>(n, x, k, distances, labels, impl, scaler);
            } else {
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
                                &nlist_visited,
                                scaler);
                    } else {
                        search_implem_10<C>(
                                i1 - i0,
                                x + i0 * d,
                                k,
                                dis_i,
                                lab_i,
                                impl,
                                &ndis,
                                &nlist_visited,
                                scaler);
                    }
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

template <class C, class Scaler>
void IndexIVFFastScan::search_implem_1(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const Scaler& scaler) const {
    FAISS_THROW_IF_NOT(orig_invlists);

    std::unique_ptr<idx_t[]> coarse_ids(new idx_t[n * nprobe]);
    std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

    quantizer->search(n, x, nprobe, coarse_dis.get(), coarse_ids.get());

    size_t dim12 = ksub * M;
    AlignedTable<float> dis_tables;
    AlignedTable<float> biases;

    compute_LUT(n, x, coarse_ids.get(), coarse_dis.get(), dis_tables, biases);

    bool single_LUT = !lookup_table_is_3d();

    size_t ndis = 0, nlist_visited = 0;

#pragma omp parallel for reduction(+ : ndis, nlist_visited)
    for (idx_t i = 0; i < n; i++) {
        int64_t* heap_ids = labels + i * k;
        float* heap_dis = distances + i * k;
        heap_heapify<C>(k, heap_dis, heap_ids);
        float* LUT = nullptr;

        if (single_LUT) {
            LUT = dis_tables.get() + i * dim12;
        }
        for (idx_t j = 0; j < nprobe; j++) {
            if (!single_LUT) {
                LUT = dis_tables.get() + (i * nprobe + j) * dim12;
            }
            idx_t list_no = coarse_ids[i * nprobe + j];
            if (list_no < 0)
                continue;
            size_t ls = orig_invlists->list_size(list_no);
            if (ls == 0)
                continue;
            InvertedLists::ScopedCodes codes(orig_invlists, list_no);
            InvertedLists::ScopedIds ids(orig_invlists, list_no);

            float bias = biases.get() ? biases[i * nprobe + j] : 0;

            estimators_from_tables_generic<C>(
                    *this,
                    codes.get(),
                    ls,
                    LUT,
                    ids.get(),
                    bias,
                    k,
                    heap_dis,
                    heap_ids,
                    scaler);
            nlist_visited++;
            ndis++;
        }
        heap_reorder<C>(k, heap_dis, heap_ids);
    }
    indexIVF_stats.nq += n;
    indexIVF_stats.ndis += ndis;
    indexIVF_stats.nlist += nlist_visited;
}

template <class C, class Scaler>
void IndexIVFFastScan::search_implem_2(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const Scaler& scaler) const {
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

    bool single_LUT = !lookup_table_is_3d();

    size_t ndis = 0, nlist_visited = 0;

#pragma omp parallel for reduction(+ : ndis, nlist_visited)
    for (idx_t i = 0; i < n; i++) {
        std::vector<uint16_t> tmp_dis(k);
        int64_t* heap_ids = labels + i * k;
        uint16_t* heap_dis = tmp_dis.data();
        heap_heapify<C>(k, heap_dis, heap_ids);
        const uint8_t* LUT = nullptr;

        if (single_LUT) {
            LUT = dis_tables.get() + i * dim12;
        }
        for (idx_t j = 0; j < nprobe; j++) {
            if (!single_LUT) {
                LUT = dis_tables.get() + (i * nprobe + j) * dim12;
            }
            idx_t list_no = coarse_ids[i * nprobe + j];
            if (list_no < 0)
                continue;
            size_t ls = orig_invlists->list_size(list_no);
            if (ls == 0)
                continue;
            InvertedLists::ScopedCodes codes(orig_invlists, list_no);
            InvertedLists::ScopedIds ids(orig_invlists, list_no);

            uint16_t bias = biases.get() ? biases[i * nprobe + j] : 0;

            estimators_from_tables_generic<C>(
                    *this,
                    codes.get(),
                    ls,
                    LUT,
                    ids.get(),
                    bias,
                    k,
                    heap_dis,
                    heap_ids,
                    scaler);

            nlist_visited++;
            ndis += ls;
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

template <class C, class Scaler>
void IndexIVFFastScan::search_implem_10(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int impl,
        size_t* ndis_out,
        size_t* nlist_out,
        const Scaler& scaler) const {
    memset(distances, -1, sizeof(float) * k * n);
    memset(labels, -1, sizeof(idx_t) * k * n);

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

    bool single_LUT = !lookup_table_is_3d();

    TIC;
    size_t ndis = 0, nlist_visited = 0;

    {
        AlignedTable<uint16_t> tmp_distances(k);
        for (idx_t i = 0; i < n; i++) {
            const uint8_t* LUT = nullptr;
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

            if (single_LUT) {
                LUT = dis_tables.get() + i * dim12;
            }
            for (idx_t j = 0; j < nprobe; j++) {
                size_t ij = i * nprobe + j;
                if (!single_LUT) {
                    LUT = dis_tables.get() + ij * dim12;
                }
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

#define DISPATCH(classHC)                                                      \
    if (dynamic_cast<classHC*>(handler.get())) {                               \
        auto* res = static_cast<classHC*>(handler.get());                      \
        pq4_accumulate_loop(                                                   \
                1, roundup(ls, bbs), bbs, M2, codes.get(), LUT, *res, scaler); \
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

template <class C, class Scaler>
void IndexIVFFastScan::search_implem_12(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int impl,
        size_t* ndis_out,
        size_t* nlist_out,
        const Scaler& scaler) const {
    if (n == 0) { // does not work well with reservoir
        return;
    }
    FAISS_THROW_IF_NOT(bbs == 32);

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

    struct QC {
        int qno;     // sequence number of the query
        int list_no; // list to visit
        int rank;    // this is the rank'th result of the coarse quantizer
    };
    bool single_LUT = !lookup_table_is_3d();

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
            lut_entries[i - i0] = single_LUT ? qc.qno : ij;
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

#define DISPATCH(classHC)                                                  \
    if (dynamic_cast<classHC*>(handler.get())) {                           \
        auto* res = static_cast<classHC*>(handler.get());                  \
        pq4_accumulate_loop_qbs(                                           \
                qbs, list_size, M2, codes.get(), LUT.get(), *res, scaler); \
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
        IVFFastScan_stats.times[i] += times[i] - times[i - 1];
    }
    IVFFastScan_stats.t_copy_pack += t_copy_pack;
    IVFFastScan_stats.t_scan += t_scan;

    if (auto* rh = dynamic_cast<ReservoirHC*>(handler.get())) {
        for (int i = 0; i < 4; i++) {
            IVFFastScan_stats.reservoir_times[i] += rh->times[i];
        }
    }

    *ndis_out = ndis;
    *nlist_out = nlist;
}

template <class C, class Scaler>
void IndexIVFFastScan::search_implem_14(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int impl,
        const Scaler& scaler) const {
    if (n == 0) { // does not work well with reservoir
        return;
    }
    FAISS_THROW_IF_NOT(bbs == 32);

    std::unique_ptr<idx_t[]> coarse_ids(new idx_t[n * nprobe]);
    std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

    uint64_t ttg0 = get_cy();

    quantizer->search(n, x, nprobe, coarse_dis.get(), coarse_ids.get());

    uint64_t ttg1 = get_cy();
    uint64_t coarse_search_tt = ttg1 - ttg0;

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

    uint64_t ttg2 = get_cy();
    uint64_t lut_compute_tt = ttg2 - ttg1;

    struct QC {
        int qno;     // sequence number of the query
        int list_no; // list to visit
        int rank;    // this is the rank'th result of the coarse quantizer
    };
    bool single_LUT = !lookup_table_is_3d();

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

    struct SE {
        size_t start; // start in the QC vector
        size_t end;   // end in the QC vector
        size_t list_size;
    };
    std::vector<SE> ses;
    size_t i0_l = 0;
    while (i0_l < qcs.size()) {
        // find all queries that access this inverted list
        int list_no = qcs[i0_l].list_no;
        size_t i1 = i0_l + 1;

        while (i1 < qcs.size() && i1 < i0_l + qbs2) {
            if (qcs[i1].list_no != list_no) {
                break;
            }
            i1++;
        }

        size_t list_size = invlists->list_size(list_no);

        if (list_size == 0) {
            i0_l = i1;
            continue;
        }
        ses.push_back(SE{i0_l, i1, list_size});
        i0_l = i1;
    }
    uint64_t ttg3 = get_cy();
    uint64_t compute_clusters_tt = ttg3 - ttg2;

    // function to handle the global heap
    using HeapForIP = CMin<float, idx_t>;
    using HeapForL2 = CMax<float, idx_t>;
    auto init_result = [&](float* simi, idx_t* idxi) {
        if (metric_type == METRIC_INNER_PRODUCT) {
            heap_heapify<HeapForIP>(k, simi, idxi);
        } else {
            heap_heapify<HeapForL2>(k, simi, idxi);
        }
    };

    auto add_local_results = [&](const float* local_dis,
                                 const idx_t* local_idx,
                                 float* simi,
                                 idx_t* idxi) {
        if (metric_type == METRIC_INNER_PRODUCT) {
            heap_addn<HeapForIP>(k, simi, idxi, local_dis, local_idx, k);
        } else {
            heap_addn<HeapForL2>(k, simi, idxi, local_dis, local_idx, k);
        }
    };

    auto reorder_result = [&](float* simi, idx_t* idxi) {
        if (metric_type == METRIC_INNER_PRODUCT) {
            heap_reorder<HeapForIP>(k, simi, idxi);
        } else {
            heap_reorder<HeapForL2>(k, simi, idxi);
        }
    };
    uint64_t ttg4 = get_cy();
    uint64_t fn_tt = ttg4 - ttg3;

    size_t ndis = 0;
    size_t nlist_visited = 0;

#pragma omp parallel reduction(+ : ndis, nlist_visited)
    {
        // storage for each thread
        std::vector<idx_t> local_idx(k * n);
        std::vector<float> local_dis(k * n);

        // prepare the result handlers
        std::unique_ptr<SIMDResultHandler<C, true>> handler;
        AlignedTable<uint16_t> tmp_distances;

        using HeapHC = HeapHandler<C, true>;
        using ReservoirHC = ReservoirHandler<C, true>;
        using SingleResultHC = SingleResultHandler<C, true>;

        if (k == 1) {
            handler.reset(new SingleResultHC(n, 0));
        } else if (impl == 14) {
            tmp_distances.resize(n * k);
            handler.reset(
                    new HeapHC(n, tmp_distances.get(), local_idx.data(), k, 0));
        } else if (impl == 15) {
            handler.reset(new ReservoirHC(n, 0, k, 2 * k));
        }

        int qbs2 = this->qbs2 ? this->qbs2 : 11;

        std::vector<uint16_t> tmp_bias;
        if (biases.get()) {
            tmp_bias.resize(qbs2);
            handler->dbias = tmp_bias.data();
        }

        uint64_t ttg5 = get_cy();
        uint64_t handler_tt = ttg5 - ttg4;

        std::set<int> q_set;
        uint64_t t_copy_pack = 0, t_scan = 0;
#pragma omp for schedule(dynamic)
        for (idx_t cluster = 0; cluster < ses.size(); cluster++) {
            uint64_t tt0 = get_cy();
            size_t i0 = ses[cluster].start;
            size_t i1 = ses[cluster].end;
            size_t list_size = ses[cluster].list_size;
            nlist_visited++;
            int list_no = qcs[i0].list_no;

            // re-organize LUTs and biases into the right order
            int nc = i1 - i0;

            std::vector<int> q_map(nc), lut_entries(nc);
            AlignedTable<uint8_t> LUT(nc * dim12);
            memset(LUT.get(), -1, nc * dim12);
            int qbs = pq4_preferred_qbs(nc);

            for (size_t i = i0; i < i1; i++) {
                const QC& qc = qcs[i];
                q_map[i - i0] = qc.qno;
                q_set.insert(qc.qno);
                int ij = qc.qno * nprobe + qc.rank;
                lut_entries[i - i0] = single_LUT ? qc.qno : ij;
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

#define DISPATCH(classHC)                                                  \
    if (dynamic_cast<classHC*>(handler.get())) {                           \
        auto* res = static_cast<classHC*>(handler.get());                  \
        pq4_accumulate_loop_qbs(                                           \
                qbs, list_size, M2, codes.get(), LUT.get(), *res, scaler); \
    }
            DISPATCH(HeapHC)
            else DISPATCH(ReservoirHC) else DISPATCH(SingleResultHC)

                    uint64_t tt2 = get_cy();
            t_copy_pack += tt1 - tt0;
            t_scan += tt2 - tt1;
        }

        // labels is in-place for HeapHC
        handler->to_flat_arrays(
                local_dis.data(),
                local_idx.data(),
                skip & 16 ? nullptr : normalizers.get());

#pragma omp single
        {
            // we init the results as a heap
            for (idx_t i = 0; i < n; i++) {
                init_result(distances + i * k, labels + i * k);
            }
        }
#pragma omp barrier
#pragma omp critical
        {
            // write to global heap  #go over only the queries
            for (std::set<int>::iterator it = q_set.begin(); it != q_set.end();
                 ++it) {
                add_local_results(
                        local_dis.data() + *it * k,
                        local_idx.data() + *it * k,
                        distances + *it * k,
                        labels + *it * k);
            }

            IVFFastScan_stats.t_copy_pack += t_copy_pack;
            IVFFastScan_stats.t_scan += t_scan;

            if (auto* rh = dynamic_cast<ReservoirHC*>(handler.get())) {
                for (int i = 0; i < 4; i++) {
                    IVFFastScan_stats.reservoir_times[i] += rh->times[i];
                }
            }
        }
#pragma omp barrier
#pragma omp single
        {
            for (idx_t i = 0; i < n; i++) {
                reorder_result(distances + i * k, labels + i * k);
            }
        }
    }

    indexIVF_stats.nq += n;
    indexIVF_stats.ndis += ndis;
    indexIVF_stats.nlist += nlist_visited;
}

void IndexIVFFastScan::reconstruct_from_offset(
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
    sa_decode(1, code.data(), recons);

    // add centroid to it
    if (by_residual) {
        std::vector<float> centroid(d);
        quantizer->reconstruct(list_no, centroid.data());
        for (int i = 0; i < d; ++i) {
            recons[i] += centroid[i];
        }
    }
}

void IndexIVFFastScan::reconstruct_orig_invlists() {
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

IVFFastScanStats IVFFastScan_stats;

template void IndexIVFFastScan::search_dispatch_implem<true, NormTableScaler>(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const NormTableScaler& scaler) const;

template void IndexIVFFastScan::search_dispatch_implem<false, NormTableScaler>(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const NormTableScaler& scaler) const;

} // namespace faiss
