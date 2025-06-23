/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

/* Function for soft heap */

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>

namespace faiss {

template <typename C>
void HeapArray<C>::heapify() {
#pragma omp parallel for
    for (int64_t j = 0; j < nh; j++)
        heap_heapify<C>(k, val + j * k, ids + j * k);
}

template <typename C>
void HeapArray<C>::reorder() {
#pragma omp parallel for
    for (int64_t j = 0; j < nh; j++)
        heap_reorder<C>(k, val + j * k, ids + j * k);
}

template <typename C>
void HeapArray<C>::addn(size_t nj, const T* vin, TI j0, size_t i0, int64_t ni) {
    if (ni == -1)
        ni = nh;
    assert(i0 >= 0 && i0 + ni <= nh);
#pragma omp parallel for if (ni * nj > 100000)
    for (int64_t i = i0; i < i0 + ni; i++) {
        T* __restrict simi = get_val(i);
        TI* __restrict idxi = get_ids(i);
        const T* ip_line = vin + (i - i0) * nj;

        for (size_t j = 0; j < nj; j++) {
            T ip = ip_line[j];
            if (C::cmp(simi[0], ip)) {
                heap_replace_top<C>(k, simi, idxi, ip, j + j0);
            }
        }
    }
}

template <typename C>
void HeapArray<C>::addn_with_ids(
        size_t nj,
        const T* vin,
        const TI* id_in,
        int64_t id_stride,
        size_t i0,
        int64_t ni) {
    if (id_in == nullptr) {
        addn(nj, vin, 0, i0, ni);
        return;
    }
    if (ni == -1)
        ni = nh;
    assert(i0 >= 0 && i0 + ni <= nh);
#pragma omp parallel for if (ni * nj > 100000)
    for (int64_t i = i0; i < i0 + ni; i++) {
        T* __restrict simi = get_val(i);
        TI* __restrict idxi = get_ids(i);
        const T* ip_line = vin + (i - i0) * nj;
        const TI* id_line = id_in + (i - i0) * id_stride;

        for (size_t j = 0; j < nj; j++) {
            T ip = ip_line[j];
            if (C::cmp(simi[0], ip)) {
                heap_replace_top<C>(k, simi, idxi, ip, id_line[j]);
            }
        }
    }
}

template <typename C>
void HeapArray<C>::addn_query_subset_with_ids(
        size_t nsubset,
        const TI* subset,
        size_t nj,
        const T* vin,
        const TI* id_in,
        int64_t id_stride) {
    FAISS_THROW_IF_NOT_MSG(id_in, "anonymous ids not supported");
    if (id_stride < 0) {
        id_stride = nj;
    }
#pragma omp parallel for if (nsubset * nj > 100000)
    for (int64_t si = 0; si < nsubset; si++) {
        TI i = subset[si];
        T* __restrict simi = get_val(i);
        TI* __restrict idxi = get_ids(i);
        const T* ip_line = vin + si * nj;
        const TI* id_line = id_in + si * id_stride;

        for (size_t j = 0; j < nj; j++) {
            T ip = ip_line[j];
            if (C::cmp(simi[0], ip)) {
                heap_replace_top<C>(k, simi, idxi, ip, id_line[j]);
            }
        }
    }
}

template <typename C>
void HeapArray<C>::per_line_extrema(T* out_val, TI* out_ids) const {
#pragma omp parallel for if (nh * k > 100000)
    for (int64_t j = 0; j < nh; j++) {
        int64_t imin = -1;
        typename C::T xval = C::Crev::neutral();
        const typename C::T* x_ = val + j * k;
        for (size_t i = 0; i < k; i++)
            if (C::cmp(x_[i], xval)) {
                xval = x_[i];
                imin = i;
            }
        if (out_val)
            out_val[j] = xval;

        if (out_ids) {
            if (ids && imin != -1)
                out_ids[j] = ids[j * k + imin];
            else
                out_ids[j] = imin;
        }
    }
}

// explicit instanciations

template struct HeapArray<CMin<float, int64_t>>;
template struct HeapArray<CMax<float, int64_t>>;
template struct HeapArray<CMin<float, int32_t>>;
template struct HeapArray<CMax<float, int32_t>>;
template struct HeapArray<CMin<int, int64_t>>;
template struct HeapArray<CMax<int, int64_t>>;

/**********************************************************
 * merge knn search results
 **********************************************************/

/** Merge result tables from several shards. The per-shard results are assumed
 * to be sorted. Note that the C comparator is reversed w.r.t. the usual top-k
 * element heap because we want the best (ie. lowest for L2) result to be on
 * top, not the worst.
 *
 * @param all_distances  size (nshard, n, k)
 * @param all_labels     size (nshard, n, k)
 * @param distances      output distances, size (n, k)
 * @param labels         output labels, size (n, k)
 */
template <class idx_t, class C>
void merge_knn_results(
        size_t n,
        size_t k,
        typename C::TI nshard,
        const typename C::T* all_distances,
        const idx_t* all_labels,
        typename C::T* distances,
        idx_t* labels) {
    using distance_t = typename C::T;
    if (k == 0) {
        return;
    }
    long stride = n * k;
#pragma omp parallel if (n * nshard * k > 100000)
    {
        std::vector<int> buf(2 * nshard);
        // index in each shard's result list
        int* pointer = buf.data();
        // (shard_ids, heap_vals): heap that indexes
        // shard -> current distance for this shard
        int* shard_ids = pointer + nshard;
        std::vector<distance_t> buf2(nshard);
        distance_t* heap_vals = buf2.data();
#pragma omp for
        for (long i = 0; i < n; i++) {
            // the heap maps values to the shard where they are
            // produced.
            const distance_t* D_in = all_distances + i * k;
            const idx_t* I_in = all_labels + i * k;
            int heap_size = 0;

            // push the first element of each shard (if not -1)
            for (long s = 0; s < nshard; s++) {
                pointer[s] = 0;
                if (I_in[stride * s] >= 0) {
                    heap_push<C>(
                            ++heap_size,
                            heap_vals,
                            shard_ids,
                            D_in[stride * s],
                            s);
                }
            }

            distance_t* D = distances + i * k;
            idx_t* I = labels + i * k;

            int j;
            for (j = 0; j < k && heap_size > 0; j++) {
                // pop element from best shard
                int s = shard_ids[0]; // top of heap
                int& p = pointer[s];
                D[j] = heap_vals[0];
                I[j] = I_in[stride * s + p];

                // pop from shard, advance pointer for this shard
                heap_pop<C>(heap_size--, heap_vals, shard_ids);
                p++;
                if (p < k && I_in[stride * s + p] >= 0) {
                    heap_push<C>(
                            ++heap_size,
                            heap_vals,
                            shard_ids,
                            D_in[stride * s + p],
                            s);
                }
            }
            for (; j < k; j++) {
                I[j] = -1;
                D[j] = C::Crev::neutral();
            }
        }
    }
}

// explicit instanciations
#define INSTANTIATE(C, distance_t)                                \
    template void merge_knn_results<int64_t, C<distance_t, int>>( \
            size_t,                                               \
            size_t,                                               \
            int,                                                  \
            const distance_t*,                                    \
            const int64_t*,                                       \
            distance_t*,                                          \
            int64_t*);

INSTANTIATE(CMin, float);
INSTANTIATE(CMax, float);
INSTANTIATE(CMin, int32_t);
INSTANTIATE(CMax, int32_t);

} // namespace faiss
