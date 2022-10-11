/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFastScan.h>

#include <limits.h>
#include <cassert>
#include <memory>

#include <omp.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/LookupTableScaler.h>
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

void IndexFastScan::init_fastscan(
        int d,
        size_t M,
        size_t nbits,
        MetricType metric,
        int bbs) {
    FAISS_THROW_IF_NOT(nbits == 4);
    FAISS_THROW_IF_NOT(bbs % 32 == 0);
    this->d = d;
    this->M = M;
    this->nbits = nbits;
    this->metric_type = metric;
    this->bbs = bbs;
    ksub = (1 << nbits);

    code_size = (M * nbits + 7) / 8;
    ntotal = ntotal2 = 0;
    M2 = roundup(M, 2);
    is_trained = false;
}

IndexFastScan::IndexFastScan()
        : bbs(0), M(0), code_size(0), ntotal2(0), M2(0) {}

void IndexFastScan::reset() {
    codes.resize(0);
    ntotal = 0;
}

void IndexFastScan::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);

    // do some blocking to avoid excessive allocs
    constexpr idx_t bs = 65536;
    if (n > bs) {
        for (idx_t i0 = 0; i0 < n; i0 += bs) {
            idx_t i1 = std::min(n, i0 + bs);
            if (verbose) {
                printf("IndexFastScan::add %zd/%zd\n", size_t(i1), size_t(n));
            }
            add(i1 - i0, x + i0 * d);
        }
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

size_t IndexFastScan::remove_ids(const IDSelector& sel) {
    idx_t j = 0;
    for (idx_t i = 0; i < ntotal; i++) {
        if (sel.is_member(i)) {
            // should be removed
        } else {
            if (i > j) {
                for (int sq = 0; sq < M; sq++) {
                    uint8_t code =
                            pq4_get_packed_element(codes.data(), bbs, M, i, sq);
                    pq4_set_packed_element(codes.data(), code, bbs, M, j, sq);
                }
            }
            j++;
        }
    }
    size_t nremove = ntotal - j;
    if (nremove > 0) {
        ntotal = j;
        ntotal2 = roundup(ntotal, bbs);
        size_t new_size = ntotal2 * M2 / 2;
        codes.resize(new_size);
    }
    return nremove;
}

void IndexFastScan::check_compatible_for_merge(const Index& otherIndex) const {
    const IndexFastScan* other =
            dynamic_cast<const IndexFastScan*>(&otherIndex);
    FAISS_THROW_IF_NOT(other);
    FAISS_THROW_IF_NOT(other->M == M);
    FAISS_THROW_IF_NOT(other->bbs == bbs);
    FAISS_THROW_IF_NOT(other->d == d);
    FAISS_THROW_IF_NOT(other->code_size == code_size);
    FAISS_THROW_IF_NOT_MSG(
            typeid(*this) == typeid(*other),
            "can only merge indexes of the same type");
}

void IndexFastScan::merge_from(Index& otherIndex, idx_t add_id) {
    check_compatible_for_merge(otherIndex);
    IndexFastScan* other = static_cast<IndexFastScan*>(&otherIndex);
    ntotal2 = roundup(ntotal + other->ntotal, bbs);
    codes.resize(ntotal2 * M2 / 2);
    for (int i = 0; i < other->ntotal; i++) {
        for (int sq = 0; sq < M; sq++) {
            uint8_t code =
                    pq4_get_packed_element(other->codes.data(), bbs, M, i, sq);
            pq4_set_packed_element(codes.data(), code, bbs, M, ntotal + i, sq);
        }
    }
    ntotal += other->ntotal;
    other->reset();
}

namespace {

template <class C, typename dis_t, class Scaler>
void estimators_from_tables_generic(
        const IndexFastScan& index,
        const uint8_t* codes,
        size_t ncodes,
        const dis_t* dis_table,
        size_t k,
        typename C::T* heap_dis,
        int64_t* heap_ids,
        const Scaler& scaler) {
    using accu_t = typename C::T;

    for (size_t j = 0; j < ncodes; ++j) {
        BitstringReader bsr(codes + j * index.code_size, index.code_size);
        accu_t dis = 0;
        const dis_t* dt = dis_table;
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
            heap_push<C>(k, heap_dis, heap_ids, dis, j);
        }
    }
}

} // anonymous namespace

using namespace quantize_lut;

void IndexFastScan::compute_quantized_LUT(
        idx_t n,
        const float* x,
        uint8_t* lut,
        float* normalizers) const {
    size_t dim12 = ksub * M;
    std::unique_ptr<float[]> dis_tables(new float[n * dim12]);
    compute_float_LUT(dis_tables.get(), n, x);

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

void IndexFastScan::search(
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

template <bool is_max, class Scaler>
void IndexFastScan::search_dispatch_implem(
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
        FAISS_THROW_MSG("not implemented");
    } else if (implem == 2 || implem == 3 || implem == 4) {
        FAISS_THROW_IF_NOT(orig_codes != nullptr);
        search_implem_234<Cfloat>(n, x, k, distances, labels, scaler);
    } else if (impl >= 12 && impl <= 15) {
        FAISS_THROW_IF_NOT(ntotal < INT_MAX);
        int nt = std::min(omp_get_max_threads(), int(n));
        if (nt < 2) {
            if (impl == 12 || impl == 13) {
                search_implem_12<C>(n, x, k, distances, labels, impl, scaler);
            } else {
                search_implem_14<C>(n, x, k, distances, labels, impl, scaler);
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
                            i1 - i0, x + i0 * d, k, dis_i, lab_i, impl, scaler);
                } else {
                    search_implem_14<C>(
                            i1 - i0, x + i0 * d, k, dis_i, lab_i, impl, scaler);
                }
            }
        }
    } else {
        FAISS_THROW_FMT("invalid implem %d impl=%d", implem, impl);
    }
}

template <class Cfloat, class Scaler>
void IndexFastScan::search_implem_234(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const Scaler& scaler) const {
    FAISS_THROW_IF_NOT(implem == 2 || implem == 3 || implem == 4);

    const size_t dim12 = ksub * M;
    std::unique_ptr<float[]> dis_tables(new float[n * dim12]);
    compute_float_LUT(dis_tables.get(), n, x);

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

        estimators_from_tables_generic<Cfloat>(
                *this,
                orig_codes,
                ntotal,
                dis_tables.get() + i * dim12,
                k,
                heap_dis,
                heap_ids,
                scaler);

        heap_reorder<Cfloat>(k, heap_dis, heap_ids);

        if (implem == 4) {
            float a = normalizers[2 * i];
            float b = normalizers[2 * i + 1];

            for (int j = 0; j < k; j++) {
                heap_dis[j] = heap_dis[j] / a + b;
            }
        }
    }
}

template <class C, class Scaler>
void IndexFastScan::search_implem_12(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int impl,
        const Scaler& scaler) const {
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
                    impl,
                    scaler);
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
                    qbs, ntotal2, M2, codes.get(), LUT.get(), handler, scaler);
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
                    qbs, ntotal2, M2, codes.get(), LUT.get(), handler, scaler);

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
                    qbs, ntotal2, M2, codes.get(), LUT.get(), handler, scaler);
        }

        if (!(skip & 8)) {
            handler.to_flat_arrays(distances, labels, normalizers.get());
        }

        FastScan_stats.t0 += handler.times[0];
        FastScan_stats.t1 += handler.times[1];
        FastScan_stats.t2 += handler.times[2];
        FastScan_stats.t3 += handler.times[3];
    }
}

FastScanStats FastScan_stats;

template <class C, class Scaler>
void IndexFastScan::search_implem_14(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int impl,
        const Scaler& scaler) const {
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
                    impl,
                    scaler);
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
                    n,
                    ntotal2,
                    bbs,
                    M2,
                    codes.get(),
                    LUT.get(),
                    handler,
                    scaler);
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
                    n,
                    ntotal2,
                    bbs,
                    M2,
                    codes.get(),
                    LUT.get(),
                    handler,
                    scaler);

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
                    n,
                    ntotal2,
                    bbs,
                    M2,
                    codes.get(),
                    LUT.get(),
                    handler,
                    scaler);
        }

        if (!(skip & 8)) {
            handler.to_flat_arrays(distances, labels, normalizers.get());
        }
    }
}

template void IndexFastScan::search_dispatch_implem<true, NormTableScaler>(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const NormTableScaler& scaler) const;

template void IndexFastScan::search_dispatch_implem<false, NormTableScaler>(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const NormTableScaler& scaler) const;

void IndexFastScan::reconstruct(idx_t key, float* recons) const {
    std::vector<uint8_t> code(code_size, 0);
    BitstringWriter bsw(code.data(), code_size);
    for (size_t m = 0; m < M; m++) {
        uint8_t c = pq4_get_packed_element(codes.data(), bbs, M2, key, m);
        bsw.write(c, nbits);
    }
    sa_decode(1, code.data(), recons);
}

} // namespace faiss
