/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <type_traits>
#include <vector>

#include <faiss/utils/Heap.h>
#include <faiss/utils/simdlib.h>

#include <faiss/impl/platform_macros.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/partitioning.h>

/** This file contains callbacks for kernels that compute distances.
 *
 * The SIMDResultHandler object is intended to be templated and inlined.
 * Methods:
 * - handle(): called when 32 distances are computed and provided in two
 *   simd16uint16. (q, b) indicate which entry it is in the block.
 * - set_block_origin(): set the sub-matrix that is being computed
 */

namespace faiss {

namespace simd_result_handlers {

/** Dummy structure that just computes a checksum on results
 * (to avoid the computation to be optimized away) */
struct DummyResultHandler {
    size_t cs = 0;

    void handle(size_t q, size_t b, simd16uint16 d0, simd16uint16 d1) {
        cs += q * 123 + b * 789 + d0.get_scalar_0() + d1.get_scalar_0();
    }

    void set_block_origin(size_t, size_t) {}
};

/** memorize results in a nq-by-nb matrix.
 *
 * j0 is the current upper-left block of the matrix
 */
struct StoreResultHandler {
    uint16_t* data;
    size_t ld; // total number of columns
    size_t i0 = 0;
    size_t j0 = 0;

    StoreResultHandler(uint16_t* data, size_t ld) : data(data), ld(ld) {}

    void handle(size_t q, size_t b, simd16uint16 d0, simd16uint16 d1) {
        size_t ofs = (q + i0) * ld + j0 + b * 32;
        d0.store(data + ofs);
        d1.store(data + ofs + 16);
    }

    void set_block_origin(size_t i0, size_t j0) {
        this->i0 = i0;
        this->j0 = j0;
    }
};

/** stores results in fixed-size matrix. */
template <int NQ, int BB>
struct FixedStorageHandler {
    simd16uint16 dis[NQ][BB];
    int i0 = 0;

    void handle(int q, int b, simd16uint16 d0, simd16uint16 d1) {
        dis[q + i0][2 * b] = d0;
        dis[q + i0][2 * b + 1] = d1;
    }

    void set_block_origin(size_t i0, size_t j0) {
        this->i0 = i0;
        assert(j0 == 0);
    }

    template <class OtherResultHandler>
    void to_other_handler(OtherResultHandler& other) const {
        for (int q = 0; q < NQ; q++) {
            for (int b = 0; b < BB; b += 2) {
                other.handle(q, b / 2, dis[q][b], dis[q][b + 1]);
            }
        }
    }
};

/** Record origin of current block  */
template <class C, bool with_id_map>
struct SIMDResultHandler {
    using TI = typename C::TI;

    bool disable = false;

    int64_t i0 = 0; // query origin
    int64_t j0 = 0; // db origin
    size_t ntotal;  // ignore excess elements after ntotal

    /// these fields are used mainly for the IVF variants (with_id_map=true)
    const TI* id_map;      // map offset in invlist to vector id
    const int* q_map;      // map q to global query
    const uint16_t* dbias; // table of biases to add to each query

    explicit SIMDResultHandler(size_t ntotal)
            : ntotal(ntotal), id_map(nullptr), q_map(nullptr), dbias(nullptr) {}

    void set_block_origin(size_t i0, size_t j0) {
        this->i0 = i0;
        this->j0 = j0;
    }

    // adjust handler data for IVF.
    void adjust_with_origin(size_t& q, simd16uint16& d0, simd16uint16& d1) {
        q += i0;

        if (dbias) {
            simd16uint16 dbias16(dbias[q]);
            d0 += dbias16;
            d1 += dbias16;
        }

        if (with_id_map) { // FIXME test on q_map instead
            q = q_map[q];
        }
    }

    // compute and adjust idx
    int64_t adjust_id(size_t b, size_t j) {
        int64_t idx = j0 + 32 * b + j;
        if (with_id_map) {
            idx = id_map[idx];
        }
        return idx;
    }

    /// return binary mask of elements below thr in (d0, d1)
    /// inverse_test returns elements above
    uint32_t get_lt_mask(
            uint16_t thr,
            size_t b,
            simd16uint16 d0,
            simd16uint16 d1) {
        simd16uint16 thr16(thr);
        uint32_t lt_mask;

        constexpr bool keep_min = C::is_max;
        if (keep_min) {
            lt_mask = ~cmp_ge32(d0, d1, thr16);
        } else {
            lt_mask = ~cmp_le32(d0, d1, thr16);
        }

        if (lt_mask == 0) {
            return 0;
        }
        uint64_t idx = j0 + b * 32;
        if (idx + 32 > ntotal) {
            if (idx >= ntotal) {
                return 0;
            }
            int nbit = (ntotal - idx);
            lt_mask &= (uint32_t(1) << nbit) - 1;
        }
        return lt_mask;
    }

    virtual void to_flat_arrays(
            float* distances,
            int64_t* labels,
            const float* normalizers = nullptr) = 0;

    virtual ~SIMDResultHandler() {}
};

/** Special version for k=1 */
template <class C, bool with_id_map = false>
struct SingleResultHandler : SIMDResultHandler<C, with_id_map> {
    using T = typename C::T;
    using TI = typename C::TI;

    struct Result {
        T val;
        TI id;
    };
    std::vector<Result> results;

    SingleResultHandler(size_t nq, size_t ntotal)
            : SIMDResultHandler<C, with_id_map>(ntotal), results(nq) {
        for (int i = 0; i < nq; i++) {
            Result res = {C::neutral(), -1};
            results[i] = res;
        }
    }

    void handle(size_t q, size_t b, simd16uint16 d0, simd16uint16 d1) {
        if (this->disable) {
            return;
        }

        this->adjust_with_origin(q, d0, d1);

        Result& res = results[q];
        uint32_t lt_mask = this->get_lt_mask(res.val, b, d0, d1);
        if (!lt_mask) {
            return;
        }

        ALIGNED(32) uint16_t d32tab[32];
        d0.store(d32tab);
        d1.store(d32tab + 16);

        while (lt_mask) {
            // find first non-zero
            int j = __builtin_ctz(lt_mask);
            lt_mask -= 1 << j;
            T dis = d32tab[j];
            if (C::cmp(res.val, dis)) {
                res.val = dis;
                res.id = this->adjust_id(b, j);
            }
        }
    }

    void to_flat_arrays(
            float* distances,
            int64_t* labels,
            const float* normalizers = nullptr) override {
        for (int q = 0; q < results.size(); q++) {
            if (!normalizers) {
                distances[q] = results[q].val;
            } else {
                float one_a = 1 / normalizers[2 * q];
                float b = normalizers[2 * q + 1];
                distances[q] = b + results[q].val * one_a;
            }
            labels[q] = results[q].id;
        }
    }
};

/** Structure that collects results in a min- or max-heap */
template <class C, bool with_id_map = false>
struct HeapHandler : SIMDResultHandler<C, with_id_map> {
    using T = typename C::T;
    using TI = typename C::TI;

    int nq;
    T* heap_dis_tab;
    TI* heap_ids_tab;

    int64_t k; // number of results to keep

    HeapHandler(
            int nq,
            T* heap_dis_tab,
            TI* heap_ids_tab,
            size_t k,
            size_t ntotal)
            : SIMDResultHandler<C, with_id_map>(ntotal),
              nq(nq),
              heap_dis_tab(heap_dis_tab),
              heap_ids_tab(heap_ids_tab),
              k(k) {
        for (int q = 0; q < nq; q++) {
            T* heap_dis_in = heap_dis_tab + q * k;
            TI* heap_ids_in = heap_ids_tab + q * k;
            heap_heapify<C>(k, heap_dis_in, heap_ids_in);
        }
    }

    void handle(size_t q, size_t b, simd16uint16 d0, simd16uint16 d1) {
        if (this->disable) {
            return;
        }

        this->adjust_with_origin(q, d0, d1);

        T* heap_dis = heap_dis_tab + q * k;
        TI* heap_ids = heap_ids_tab + q * k;

        uint16_t cur_thresh =
                heap_dis[0] < 65536 ? (uint16_t)(heap_dis[0]) : 0xffff;

        // here we handle the reverse comparison case as well
        uint32_t lt_mask = this->get_lt_mask(cur_thresh, b, d0, d1);

        if (!lt_mask) {
            return;
        }

        ALIGNED(32) uint16_t d32tab[32];
        d0.store(d32tab);
        d1.store(d32tab + 16);

        while (lt_mask) {
            // find first non-zero
            int j = __builtin_ctz(lt_mask);
            lt_mask -= 1 << j;
            T dis = d32tab[j];
            if (C::cmp(heap_dis[0], dis)) {
                int64_t idx = this->adjust_id(b, j);
                heap_pop<C>(k, heap_dis, heap_ids);
                heap_push<C>(k, heap_dis, heap_ids, dis, idx);
            }
        }
    }

    void to_flat_arrays(
            float* distances,
            int64_t* labels,
            const float* normalizers = nullptr) override {
        for (int q = 0; q < nq; q++) {
            T* heap_dis_in = heap_dis_tab + q * k;
            TI* heap_ids_in = heap_ids_tab + q * k;
            heap_reorder<C>(k, heap_dis_in, heap_ids_in);
            int64_t* heap_ids = labels + q * k;
            float* heap_dis = distances + q * k;

            float one_a = 1.0, b = 0.0;
            if (normalizers) {
                one_a = 1 / normalizers[2 * q];
                b = normalizers[2 * q + 1];
            }
            for (int j = 0; j < k; j++) {
                heap_ids[j] = heap_ids_in[j];
                heap_dis[j] = heap_dis_in[j] * one_a + b;
            }
        }
    }
};

/** Simple top-N implementation using a reservoir.
 *
 * Results are stored when they are below the threshold until the capacity is
 * reached. Then a partition sort is used to update the threshold. */

namespace {

uint64_t get_cy() {
#ifdef MICRO_BENCHMARK
    uint32_t high, low;
    asm volatile("rdtsc \n\t" : "=a"(low), "=d"(high));
    return ((uint64_t)high << 32) | (low);
#else
    return 0;
#endif
}

} // anonymous namespace

template <class C>
struct ReservoirTopN {
    using T = typename C::T;
    using TI = typename C::TI;

    T* vals;
    TI* ids;

    size_t i;        // number of stored elements
    size_t n;        // number of requested elements
    size_t capacity; // size of storage
    size_t cycles = 0;

    T threshold; // current threshold

    ReservoirTopN(size_t n, size_t capacity, T* vals, TI* ids)
            : vals(vals), ids(ids), i(0), n(n), capacity(capacity) {
        assert(n < capacity);
        threshold = C::neutral();
    }

    void add(T val, TI id) {
        if (C::cmp(threshold, val)) {
            if (i == capacity) {
                shrink_fuzzy();
            }
            vals[i] = val;
            ids[i] = id;
            i++;
        }
    }

    /// shrink number of stored elements to n
    void shrink_xx() {
        uint64_t t0 = get_cy();
        qselect(vals, ids, i, n);
        i = n; // forget all elements above i = n
        threshold = C::Crev::neutral();
        for (size_t j = 0; j < n; j++) {
            if (C::cmp(vals[j], threshold)) {
                threshold = vals[j];
            }
        }
        cycles += get_cy() - t0;
    }

    void shrink() {
        uint64_t t0 = get_cy();
        threshold = partition<C>(vals, ids, i, n);
        i = n;
        cycles += get_cy() - t0;
    }

    void shrink_fuzzy() {
        uint64_t t0 = get_cy();
        assert(i == capacity);
        threshold = partition_fuzzy<C>(
                vals, ids, capacity, n, (capacity + n) / 2, &i);
        cycles += get_cy() - t0;
    }
};

/** Handler built from several ReservoirTopN (one per query) */
template <class C, bool with_id_map = false>
struct ReservoirHandler : SIMDResultHandler<C, with_id_map> {
    using T = typename C::T;
    using TI = typename C::TI;

    size_t capacity; // rounded up to multiple of 16
    std::vector<TI> all_ids;
    AlignedTable<T> all_vals;

    std::vector<ReservoirTopN<C>> reservoirs;

    uint64_t times[4];

    ReservoirHandler(size_t nq, size_t ntotal, size_t n, size_t capacity_in)
            : SIMDResultHandler<C, with_id_map>(ntotal),
              capacity((capacity_in + 15) & ~15),
              all_ids(nq * capacity),
              all_vals(nq * capacity) {
        assert(capacity % 16 == 0);
        for (size_t i = 0; i < nq; i++) {
            reservoirs.emplace_back(
                    n,
                    capacity,
                    all_vals.get() + i * capacity,
                    all_ids.data() + i * capacity);
        }
        times[0] = times[1] = times[2] = times[3] = 0;
    }

    void handle(size_t q, size_t b, simd16uint16 d0, simd16uint16 d1) {
        uint64_t t0 = get_cy();
        if (this->disable) {
            return;
        }
        this->adjust_with_origin(q, d0, d1);

        ReservoirTopN<C>& res = reservoirs[q];
        uint32_t lt_mask = this->get_lt_mask(res.threshold, b, d0, d1);
        uint64_t t1 = get_cy();
        times[0] += t1 - t0;

        if (!lt_mask) {
            return;
        }
        ALIGNED(32) uint16_t d32tab[32];
        d0.store(d32tab);
        d1.store(d32tab + 16);

        while (lt_mask) {
            // find first non-zero
            int j = __builtin_ctz(lt_mask);
            lt_mask -= 1 << j;
            T dis = d32tab[j];
            res.add(dis, this->adjust_id(b, j));
        }
        times[1] += get_cy() - t1;
    }

    void to_flat_arrays(
            float* distances,
            int64_t* labels,
            const float* normalizers = nullptr) override {
        using Cf = typename std::conditional<
                C::is_max,
                CMax<float, int64_t>,
                CMin<float, int64_t>>::type;

        uint64_t t0 = get_cy();
        uint64_t t3 = 0;
        std::vector<int> perm(reservoirs[0].n);
        for (int q = 0; q < reservoirs.size(); q++) {
            ReservoirTopN<C>& res = reservoirs[q];
            size_t n = res.n;

            if (res.i > res.n) {
                res.shrink();
            }
            int64_t* heap_ids = labels + q * n;
            float* heap_dis = distances + q * n;

            float one_a = 1.0, b = 0.0;
            if (normalizers) {
                one_a = 1 / normalizers[2 * q];
                b = normalizers[2 * q + 1];
            }
            for (int i = 0; i < res.i; i++) {
                perm[i] = i;
            }
            // indirect sort of result arrays
            std::sort(perm.begin(), perm.begin() + res.i, [&res](int i, int j) {
                return C::cmp(res.vals[j], res.vals[i]);
            });
            for (int i = 0; i < res.i; i++) {
                heap_dis[i] = res.vals[perm[i]] * one_a + b;
                heap_ids[i] = res.ids[perm[i]];
            }

            // possibly add empty results
            heap_heapify<Cf>(n - res.i, heap_dis + res.i, heap_ids + res.i);

            t3 += res.cycles;
        }
        times[2] += get_cy() - t0;
        times[3] += t3;
    }
};

} // namespace simd_result_handlers

} // namespace faiss
