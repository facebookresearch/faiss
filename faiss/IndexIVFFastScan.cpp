/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
    // unlike other indexes, we prefer no residuals for performance reasons.
    by_residual = false;
    FAISS_THROW_IF_NOT(metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT);
}

IndexIVFFastScan::IndexIVFFastScan() {
    bbs = 0;
    M2 = 0;
    is_trained = false;
    by_residual = false;
}

void IndexIVFFastScan::init_fastscan(
        size_t M,
        size_t nbits_init,
        size_t nlist,
        MetricType /* metric */,
        int bbs_2) {
    FAISS_THROW_IF_NOT(bbs_2 % 32 == 0);
    FAISS_THROW_IF_NOT(nbits_init == 4);

    this->M = M;
    this->nbits = nbits_init;
    this->bbs = bbs_2;
    ksub = (1 << nbits_init);
    M2 = roundup(M, 2);
    code_size = M2 / 2;

    is_trained = false;
    replace_invlists(new BlockInvertedLists(nlist, get_CodePacker()), true);
}

void IndexIVFFastScan::init_code_packer() {
    auto bil = dynamic_cast<BlockInvertedLists*>(invlists);
    FAISS_THROW_IF_NOT(bil);
    delete bil->packer; // in case there was one before
    bil->packer = get_CodePacker();
}

IndexIVFFastScan::~IndexIVFFastScan() = default;

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

    direct_map.check_can_add(xids);
    std::unique_ptr<idx_t[]> idx(new idx_t[n]);
    quantizer->assign(n, x, idx.get());

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

CodePacker* IndexIVFFastScan::get_CodePacker() const {
    return new CodePackerPQ4(M, bbs);
}

/*********************************************************
 * search
 *********************************************************/

namespace {

template <class C, typename dis_t>
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
        const NormTableScaler* scaler) {
    using accu_t = typename C::T;
    size_t nscale = scaler ? scaler->nscale : 0;
    for (size_t j = 0; j < ncodes; ++j) {
        BitstringReader bsr(codes + j * index.code_size, index.code_size);
        accu_t dis = bias;
        const dis_t* __restrict dt = dis_table;

        for (size_t m = 0; m < index.M - nscale; m++) {
            uint64_t c = bsr.read(index.nbits);
            dis += dt[c];
            dt += index.ksub;
        }

        if (scaler) {
            for (size_t m = 0; m < nscale; m++) {
                uint64_t c = bsr.read(index.nbits);
                dis += scaler->scale_one(dt[c]);
                dt += index.ksub;
            }
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
        const CoarseQuantized& cq,
        AlignedTable<uint8_t>& dis_tables,
        AlignedTable<uint16_t>& biases,
        float* normalizers) const {
    AlignedTable<float> dis_tables_float;
    AlignedTable<float> biases_float;

    compute_LUT(n, x, cq, dis_tables_float, biases_float);
    size_t nprobe = cq.nprobe;
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

    // OMP for MSVC requires i to have signed integral type
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
        const SearchParameters* params_in) const {
    const IVFSearchParameters* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters*>(params_in);
        FAISS_THROW_IF_NOT_MSG(
                params, "IndexIVFFastScan params have incorrect type");
    }

    search_preassigned(
            n, x, k, nullptr, nullptr, distances, labels, false, params);
}

void IndexIVFFastScan::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* assign,
        const float* centroid_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats) const {
    size_t nprobe = this->nprobe;
    if (params) {
        FAISS_THROW_IF_NOT(params->max_codes == 0);
        nprobe = params->nprobe;
    }

    FAISS_THROW_IF_NOT_MSG(
            !store_pairs, "store_pairs not supported for this index");
    FAISS_THROW_IF_NOT_MSG(!stats, "stats not supported for this index");
    FAISS_THROW_IF_NOT(k > 0);

    const CoarseQuantized cq = {nprobe, centroid_dis, assign};
    search_dispatch_implem(n, x, k, distances, labels, cq, nullptr, params);
}

void IndexIVFFastScan::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params_in) const {
    size_t nprobe = this->nprobe;
    const IVFSearchParameters* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters*>(params_in);
        FAISS_THROW_IF_NOT_MSG(
                params, "IndexIVFFastScan params have incorrect type");
        nprobe = params->nprobe;
    }

    const CoarseQuantized cq = {nprobe, nullptr, nullptr};
    range_search_dispatch_implem(n, x, radius, *result, cq, nullptr, params);
}

namespace {

template <class C>
ResultHandlerCompare<C, true>* make_knn_handler_fixC(
        int impl,
        idx_t n,
        idx_t k,
        float* distances,
        idx_t* labels,
        const IDSelector* sel) {
    using HeapHC = HeapHandler<C, true>;
    using ReservoirHC = ReservoirHandler<C, true>;
    using SingleResultHC = SingleResultHandler<C, true>;

    if (k == 1) {
        return new SingleResultHC(n, 0, distances, labels, sel);
    } else if (impl % 2 == 0) {
        return new HeapHC(n, 0, k, distances, labels, sel);
    } else /* if (impl % 2 == 1) */ {
        return new ReservoirHC(n, 0, k, 2 * k, distances, labels, sel);
    }
}

SIMDResultHandlerToFloat* make_knn_handler(
        bool is_max,
        int impl,
        idx_t n,
        idx_t k,
        float* distances,
        idx_t* labels,
        const IDSelector* sel) {
    if (is_max) {
        return make_knn_handler_fixC<CMax<uint16_t, int64_t>>(
                impl, n, k, distances, labels, sel);
    } else {
        return make_knn_handler_fixC<CMin<uint16_t, int64_t>>(
                impl, n, k, distances, labels, sel);
    }
}

using CoarseQuantized = IndexIVFFastScan::CoarseQuantized;

struct CoarseQuantizedWithBuffer : CoarseQuantized {
    explicit CoarseQuantizedWithBuffer(const CoarseQuantized& cq)
            : CoarseQuantized(cq) {}

    bool done() const {
        return ids != nullptr;
    }

    std::vector<idx_t> ids_buffer;
    std::vector<float> dis_buffer;

    void quantize(
            const Index* quantizer,
            idx_t n,
            const float* x,
            const SearchParameters* quantizer_params) {
        dis_buffer.resize(nprobe * n);
        ids_buffer.resize(nprobe * n);
        quantizer->search(
                n,
                x,
                nprobe,
                dis_buffer.data(),
                ids_buffer.data(),
                quantizer_params);
        dis = dis_buffer.data();
        ids = ids_buffer.data();
    }
};

struct CoarseQuantizedSlice : CoarseQuantizedWithBuffer {
    size_t i0, i1;
    CoarseQuantizedSlice(const CoarseQuantized& cq, size_t i0, size_t i1)
            : CoarseQuantizedWithBuffer(cq), i0(i0), i1(i1) {
        if (done()) {
            dis += nprobe * i0;
            ids += nprobe * i0;
        }
    }

    void quantize_slice(
            const Index* quantizer,
            const float* x,
            const SearchParameters* quantizer_params) {
        quantize(quantizer, i1 - i0, x + quantizer->d * i0, quantizer_params);
    }
};

int compute_search_nslice(
        const IndexIVFFastScan* index,
        size_t n,
        size_t nprobe) {
    int nslice;
    if (n <= omp_get_max_threads()) {
        nslice = n;
    } else if (index->lookup_table_is_3d()) {
        // make sure we don't make too big LUT tables
        size_t lut_size_per_query = index->M * index->ksub * nprobe *
                (sizeof(float) + sizeof(uint8_t));

        size_t max_lut_size = precomputed_table_max_bytes;
        // how many queries we can handle within mem budget
        size_t nq_ok = std::max(max_lut_size / lut_size_per_query, size_t(1));
        nslice = roundup(
                std::max(size_t(n / nq_ok), size_t(1)), omp_get_max_threads());
    } else {
        // LUTs unlikely to be a limiting factor
        nslice = omp_get_max_threads();
    }
    return nslice;
}

} // namespace

void IndexIVFFastScan::search_dispatch_implem(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const CoarseQuantized& cq_in,
        const NormTableScaler* scaler,
        const IVFSearchParameters* params) const {
    const idx_t nprobe = params ? params->nprobe : this->nprobe;
    const IDSelector* sel = (params) ? params->sel : nullptr;
    const SearchParameters* quantizer_params =
            params ? params->quantizer_params : nullptr;

    bool is_max = !is_similarity_metric(metric_type);
    using RH = SIMDResultHandlerToFloat;

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
        if (k > 20) { // use reservoir rather than heap
            impl++;
        }
    }

    bool multiple_threads =
            n > 1 && impl >= 10 && impl <= 13 && omp_get_max_threads() > 1;
    if (impl >= 100) {
        multiple_threads = false;
        impl -= 100;
    }

    CoarseQuantizedWithBuffer cq(cq_in);
    cq.nprobe = nprobe;

    if (!cq.done() && !multiple_threads) {
        // we do the coarse quantization here execpt when search is
        // sliced over threads (then it is more efficient to have each thread do
        // its own coarse quantization)
        cq.quantize(quantizer, n, x, quantizer_params);
        invlists->prefetch_lists(cq.ids, n * cq.nprobe);
    }

    if (impl == 1) {
        if (is_max) {
            search_implem_1<CMax<float, int64_t>>(
                    n, x, k, distances, labels, cq, scaler, params);
        } else {
            search_implem_1<CMin<float, int64_t>>(
                    n, x, k, distances, labels, cq, scaler, params);
        }
    } else if (impl == 2) {
        if (is_max) {
            search_implem_2<CMax<uint16_t, int64_t>>(
                    n, x, k, distances, labels, cq, scaler, params);
        } else {
            search_implem_2<CMin<uint16_t, int64_t>>(
                    n, x, k, distances, labels, cq, scaler, params);
        }
    } else if (impl >= 10 && impl <= 15) {
        size_t ndis = 0, nlist_visited = 0;

        if (!multiple_threads) {
            // clang-format off
            if (impl == 12 || impl == 13) {
                std::unique_ptr<RH> handler(
                    make_knn_handler(
                        is_max, 
                        impl, 
                        n, 
                        k, 
                        distances, 
                        labels, sel
                    )
                );
                search_implem_12(
                        n, x, *handler.get(),
                        cq, &ndis, &nlist_visited, scaler, params);
            } else if (impl == 14 || impl == 15) {
                search_implem_14(
                        n, x, k, distances, labels,
                        cq, impl, scaler, params);
            } else {
                std::unique_ptr<RH> handler(
                    make_knn_handler(
                        is_max, 
                        impl, 
                        n, 
                        k, 
                        distances, 
                        labels,
                        sel
                    )
                );
                search_implem_10(
                        n, x, *handler.get(), cq,
                        &ndis, &nlist_visited, scaler, params);
            }
            // clang-format on
        } else {
            // explicitly slice over threads
            int nslice = compute_search_nslice(this, n, cq.nprobe);
            if (impl == 14 || impl == 15) {
                // this might require slicing if there are too
                // many queries (for now we keep this simple)
                search_implem_14(
                        n, x, k, distances, labels, cq, impl, scaler, params);
            } else {
#pragma omp parallel for reduction(+ : ndis, nlist_visited)
                for (int slice = 0; slice < nslice; slice++) {
                    idx_t i0 = n * slice / nslice;
                    idx_t i1 = n * (slice + 1) / nslice;
                    float* dis_i = distances + i0 * k;
                    idx_t* lab_i = labels + i0 * k;
                    CoarseQuantizedSlice cq_i(cq, i0, i1);
                    if (!cq_i.done()) {
                        cq_i.quantize_slice(quantizer, x, quantizer_params);
                    }
                    std::unique_ptr<RH> handler(make_knn_handler(
                            is_max, impl, i1 - i0, k, dis_i, lab_i, sel));
                    // clang-format off
                    if (impl == 12 || impl == 13) {
                        search_implem_12(
                                i1 - i0, x + i0 * d, *handler.get(),
                                cq_i, &ndis, &nlist_visited, scaler, params);
                    } else {
                        search_implem_10(
                                i1 - i0, x + i0 * d, *handler.get(),
                                cq_i, &ndis, &nlist_visited, scaler, params);
                    }
                    // clang-format on
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

void IndexIVFFastScan::range_search_dispatch_implem(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult& rres,
        const CoarseQuantized& cq_in,
        const NormTableScaler* scaler,
        const IVFSearchParameters* params) const {
    // const idx_t nprobe = params ? params->nprobe : this->nprobe;
    const IDSelector* sel = (params) ? params->sel : nullptr;
    const SearchParameters* quantizer_params =
            params ? params->quantizer_params : nullptr;

    bool is_max = !is_similarity_metric(metric_type);

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
    }

    CoarseQuantizedWithBuffer cq(cq_in);

    bool multiple_threads =
            n > 1 && impl >= 10 && impl <= 13 && omp_get_max_threads() > 1;
    if (impl >= 100) {
        multiple_threads = false;
        impl -= 100;
    }

    if (!multiple_threads && !cq.done()) {
        cq.quantize(quantizer, n, x, quantizer_params);
        invlists->prefetch_lists(cq.ids, n * cq.nprobe);
    }

    size_t ndis = 0, nlist_visited = 0;

    if (!multiple_threads) { // single thread
        std::unique_ptr<SIMDResultHandlerToFloat> handler;
        if (is_max) {
            handler.reset(new RangeHandler<CMax<uint16_t, int64_t>, true>(
                    rres, radius, 0, sel));
        } else {
            handler.reset(new RangeHandler<CMin<uint16_t, int64_t>, true>(
                    rres, radius, 0, sel));
        }
        if (impl == 12) {
            search_implem_12(
                    n, x, *handler.get(), cq, &ndis, &nlist_visited, scaler);
        } else if (impl == 10) {
            search_implem_10(
                    n, x, *handler.get(), cq, &ndis, &nlist_visited, scaler);
        } else {
            FAISS_THROW_FMT("Range search implem %d not implemented", impl);
        }
    } else {
        // explicitly slice over threads
        int nslice = compute_search_nslice(this, n, cq.nprobe);
#pragma omp parallel
        {
            RangeSearchPartialResult pres(&rres);

#pragma omp for reduction(+ : ndis, nlist_visited)
            for (int slice = 0; slice < nslice; slice++) {
                idx_t i0 = n * slice / nslice;
                idx_t i1 = n * (slice + 1) / nslice;
                CoarseQuantizedSlice cq_i(cq, i0, i1);
                if (!cq_i.done()) {
                    cq_i.quantize_slice(quantizer, x, quantizer_params);
                }
                std::unique_ptr<SIMDResultHandlerToFloat> handler;
                if (is_max) {
                    handler.reset(new PartialRangeHandler<
                                  CMax<uint16_t, int64_t>,
                                  true>(pres, radius, 0, i0, i1, sel));
                } else {
                    handler.reset(new PartialRangeHandler<
                                  CMin<uint16_t, int64_t>,
                                  true>(pres, radius, 0, i0, i1, sel));
                }

                if (impl == 12 || impl == 13) {
                    search_implem_12(
                            i1 - i0,
                            x + i0 * d,
                            *handler.get(),
                            cq_i,
                            &ndis,
                            &nlist_visited,
                            scaler,
                            params);
                } else {
                    search_implem_10(
                            i1 - i0,
                            x + i0 * d,
                            *handler.get(),
                            cq_i,
                            &ndis,
                            &nlist_visited,
                            scaler,
                            params);
                }
            }
            pres.finalize();
        }
    }

    indexIVF_stats.nq += n;
    indexIVF_stats.ndis += ndis;
    indexIVF_stats.nlist += nlist_visited;
}

template <class C>
void IndexIVFFastScan::search_implem_1(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const CoarseQuantized& cq,
        const NormTableScaler* scaler,
        const IVFSearchParameters* params) const {
    FAISS_THROW_IF_NOT(orig_invlists);

    size_t dim12 = ksub * M;
    AlignedTable<float> dis_tables;
    AlignedTable<float> biases;

    compute_LUT(n, x, cq, dis_tables, biases);

    bool single_LUT = !lookup_table_is_3d();

    size_t ndis = 0, nlist_visited = 0;
    size_t nprobe = cq.nprobe;
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
            idx_t list_no = cq.ids[i * nprobe + j];
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

template <class C>
void IndexIVFFastScan::search_implem_2(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const CoarseQuantized& cq,
        const NormTableScaler* scaler,
        const IVFSearchParameters* params) const {
    FAISS_THROW_IF_NOT(orig_invlists);

    size_t dim12 = ksub * M2;
    AlignedTable<uint8_t> dis_tables;
    AlignedTable<uint16_t> biases;
    std::unique_ptr<float[]> normalizers(new float[2 * n]);

    compute_LUT_uint8(n, x, cq, dis_tables, biases, normalizers.get());

    bool single_LUT = !lookup_table_is_3d();

    size_t ndis = 0, nlist_visited = 0;
    size_t nprobe = cq.nprobe;

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
            idx_t list_no = cq.ids[i * nprobe + j];
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

void IndexIVFFastScan::search_implem_10(
        idx_t n,
        const float* x,
        SIMDResultHandlerToFloat& handler,
        const CoarseQuantized& cq,
        size_t* ndis_out,
        size_t* nlist_out,
        const NormTableScaler* scaler,
        const IVFSearchParameters* params) const {
    size_t dim12 = ksub * M2;
    AlignedTable<uint8_t> dis_tables;
    AlignedTable<uint16_t> biases;
    std::unique_ptr<float[]> normalizers(new float[2 * n]);

    compute_LUT_uint8(n, x, cq, dis_tables, biases, normalizers.get());

    bool single_LUT = !lookup_table_is_3d();

    size_t ndis = 0;
    int qmap1[1];

    handler.q_map = qmap1;
    handler.begin(skip & 16 ? nullptr : normalizers.get());
    size_t nprobe = cq.nprobe;

    for (idx_t i = 0; i < n; i++) {
        const uint8_t* LUT = nullptr;
        qmap1[0] = i;

        if (single_LUT) {
            LUT = dis_tables.get() + i * dim12;
        }
        for (idx_t j = 0; j < nprobe; j++) {
            size_t ij = i * nprobe + j;
            if (!single_LUT) {
                LUT = dis_tables.get() + ij * dim12;
            }
            if (biases.get()) {
                handler.dbias = biases.get() + ij;
            }

            idx_t list_no = cq.ids[ij];
            if (list_no < 0) {
                continue;
            }
            size_t ls = invlists->list_size(list_no);
            if (ls == 0) {
                continue;
            }

            InvertedLists::ScopedCodes codes(invlists, list_no);
            InvertedLists::ScopedIds ids(invlists, list_no);

            handler.ntotal = ls;
            handler.id_map = ids.get();

            pq4_accumulate_loop(
                    1,
                    roundup(ls, bbs),
                    bbs,
                    M2,
                    codes.get(),
                    LUT,
                    handler,
                    scaler);

            ndis++;
        }
    }

    handler.end();
    *ndis_out = ndis;
    *nlist_out = nlist;
}

void IndexIVFFastScan::search_implem_12(
        idx_t n,
        const float* x,
        SIMDResultHandlerToFloat& handler,
        const CoarseQuantized& cq,
        size_t* ndis_out,
        size_t* nlist_out,
        const NormTableScaler* scaler,
        const IVFSearchParameters* params) const {
    if (n == 0) { // does not work well with reservoir
        return;
    }
    FAISS_THROW_IF_NOT(bbs == 32);

    size_t dim12 = ksub * M2;
    AlignedTable<uint8_t> dis_tables;
    AlignedTable<uint16_t> biases;
    std::unique_ptr<float[]> normalizers(new float[2 * n]);

    compute_LUT_uint8(n, x, cq, dis_tables, biases, normalizers.get());

    handler.begin(skip & 16 ? nullptr : normalizers.get());

    struct QC {
        int qno;     // sequence number of the query
        int list_no; // list to visit
        int rank;    // this is the rank'th result of the coarse quantizer
    };
    bool single_LUT = !lookup_table_is_3d();
    size_t nprobe = cq.nprobe;

    std::vector<QC> qcs;
    {
        int ij = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nprobe; j++) {
                if (cq.ids[ij] >= 0) {
                    qcs.push_back(QC{i, int(cq.ids[ij]), int(j)});
                }
                ij++;
            }
        }
        std::sort(qcs.begin(), qcs.end(), [](const QC& a, const QC& b) {
            return a.list_no < b.list_no;
        });
    }

    // prepare the result handlers

    int actual_qbs2 = this->qbs2 ? this->qbs2 : 11;

    std::vector<uint16_t> tmp_bias;
    if (biases.get()) {
        tmp_bias.resize(actual_qbs2);
        handler.dbias = tmp_bias.data();
    }

    size_t ndis = 0;

    size_t i0 = 0;
    uint64_t t_copy_pack = 0, t_scan = 0;
    while (i0 < qcs.size()) {
        // find all queries that access this inverted list
        int list_no = qcs[i0].list_no;
        size_t i1 = i0 + 1;

        while (i1 < qcs.size() && i1 < i0 + actual_qbs2) {
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
        int qbs_for_list = pq4_preferred_qbs(nc);

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
                qbs_for_list,
                M2,
                dis_tables.get(),
                lut_entries.data(),
                LUT.get());

        // access the inverted list

        ndis += (i1 - i0) * list_size;

        InvertedLists::ScopedCodes codes(invlists, list_no);
        InvertedLists::ScopedIds ids(invlists, list_no);

        // prepare the handler

        handler.ntotal = list_size;
        handler.q_map = q_map.data();
        handler.id_map = ids.get();

        pq4_accumulate_loop_qbs(
                qbs_for_list,
                list_size,
                M2,
                codes.get(),
                LUT.get(),
                handler,
                scaler);
        // prepare for next loop
        i0 = i1;
    }

    handler.end();

    // these stats are not thread-safe

    IVFFastScan_stats.t_copy_pack += t_copy_pack;
    IVFFastScan_stats.t_scan += t_scan;

    *ndis_out = ndis;
    *nlist_out = nlist;
}

void IndexIVFFastScan::search_implem_14(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const CoarseQuantized& cq,
        int impl,
        const NormTableScaler* scaler,
        const IVFSearchParameters* params) const {
    if (n == 0) { // does not work well with reservoir
        return;
    }
    FAISS_THROW_IF_NOT(bbs == 32);

    const IDSelector* sel = params ? params->sel : nullptr;

    size_t dim12 = ksub * M2;
    AlignedTable<uint8_t> dis_tables;
    AlignedTable<uint16_t> biases;
    std::unique_ptr<float[]> normalizers(new float[2 * n]);

    compute_LUT_uint8(n, x, cq, dis_tables, biases, normalizers.get());

    struct QC {
        int qno;     // sequence number of the query
        int list_no; // list to visit
        int rank;    // this is the rank'th result of the coarse quantizer
    };
    bool single_LUT = !lookup_table_is_3d();
    size_t nprobe = cq.nprobe;

    std::vector<QC> qcs;
    {
        int ij = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nprobe; j++) {
                if (cq.ids[ij] >= 0) {
                    qcs.push_back(QC{i, int(cq.ids[ij]), int(j)});
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

    // function to handle the global heap
    bool is_max = !is_similarity_metric(metric_type);
    using HeapForIP = CMin<float, idx_t>;
    using HeapForL2 = CMax<float, idx_t>;
    auto init_result = [&](float* simi, idx_t* idxi) {
        if (!is_max) {
            heap_heapify<HeapForIP>(k, simi, idxi);
        } else {
            heap_heapify<HeapForL2>(k, simi, idxi);
        }
    };

    auto add_local_results = [&](const float* local_dis,
                                 const idx_t* local_idx,
                                 float* simi,
                                 idx_t* idxi) {
        if (!is_max) {
            heap_addn<HeapForIP>(k, simi, idxi, local_dis, local_idx, k);
        } else {
            heap_addn<HeapForL2>(k, simi, idxi, local_dis, local_idx, k);
        }
    };

    auto reorder_result = [&](float* simi, idx_t* idxi) {
        if (!is_max) {
            heap_reorder<HeapForIP>(k, simi, idxi);
        } else {
            heap_reorder<HeapForL2>(k, simi, idxi);
        }
    };

    size_t ndis = 0;
    size_t nlist_visited = 0;

#pragma omp parallel reduction(+ : ndis, nlist_visited)
    {
        // storage for each thread
        std::vector<idx_t> local_idx(k * n);
        std::vector<float> local_dis(k * n);

        // prepare the result handlers
        std::unique_ptr<SIMDResultHandlerToFloat> handler(make_knn_handler(
                is_max, impl, n, k, local_dis.data(), local_idx.data(), sel));
        handler->begin(normalizers.get());

        int actual_qbs2 = this->qbs2 ? this->qbs2 : 11;

        std::vector<uint16_t> tmp_bias;
        if (biases.get()) {
            tmp_bias.resize(actual_qbs2);
            handler->dbias = tmp_bias.data();
        }

        std::set<int> q_set;
        uint64_t t_copy_pack = 0, t_scan = 0;
#pragma omp for schedule(dynamic)
        for (idx_t cluster = 0; cluster < ses.size(); cluster++) {
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
            int qbs_for_list = pq4_preferred_qbs(nc);

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
                    qbs_for_list,
                    M2,
                    dis_tables.get(),
                    lut_entries.data(),
                    LUT.get());

            // access the inverted list

            ndis += (i1 - i0) * list_size;

            InvertedLists::ScopedCodes codes(invlists, list_no);
            InvertedLists::ScopedIds ids(invlists, list_no);

            // prepare the handler

            handler->ntotal = list_size;
            handler->q_map = q_map.data();
            handler->id_map = ids.get();

            pq4_accumulate_loop_qbs(
                    qbs_for_list,
                    list_size,
                    M2,
                    codes.get(),
                    LUT.get(),
                    *handler.get(),
                    scaler);
        }

        // labels is in-place for HeapHC
        handler->end();

        // merge per-thread results
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

} // namespace faiss
