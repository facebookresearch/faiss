/**
 * Copyright (c) Facebook, Inc. and its affiliates.
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


/* For One Attribute */
template <typename C>
void HeapArrayOneAttribute<C>::heapify() {
#pragma omp parallel for
    for (int64_t j = 0; j < nh; j++)
        heap_heapify_one_attribute<C>(k, val + j * k, ids + j * k, attr + j * k);
}

template <typename C>
void HeapArrayOneAttribute<C>::reorder() {
#pragma omp parallel for
    for (int64_t j = 0; j < nh; j++)
        heap_reorder_one_attribute<C>(k, val + j * k, ids + j * k, attr + j * k);
}

template <typename C>
void HeapArrayOneAttribute<C>::addn(size_t nj, const T* vin, const T* atrin, TI j0, size_t i0, int64_t ni) {
    if (ni == -1)
        ni = nh;
    assert(i0 >= 0 && i0 + ni <= nh);
#pragma omp parallel for if (ni * nj > 100000)
    for (int64_t i = i0; i < i0 + ni; i++) {
        T* __restrict simi = get_val(i);
        TI* __restrict idxi = get_ids(i);
        T* __restrict attri = get_attr(i);

        const T* ip_line = vin + (i - i0) * nj;
        const T* atrin_line = atrin + (i - i0) * nj;

        for (size_t j = 0; j < nj; j++) {
            T ip = ip_line[j];
            T atrin_tmp = atrin_line[j];
            if (C::cmp(simi[0], ip)) {
                heap_replace_top_one_attribute<C>(k, simi, idxi, attri, ip, j + j0, atrin_tmp);
            }
        }
    }
}

template <typename C>
void HeapArrayOneAttribute<C>::addn_with_ids(
        size_t nj,
        const T* vin,
        const T* atrin,
        const TI* id_in,
        int64_t id_stride,
        size_t i0,
        int64_t ni) {
    if (id_in == nullptr) {
        addn(nj, vin, atrin, 0, i0, ni);
        return;
    }
    if (ni == -1)
        ni = nh;
    assert(i0 >= 0 && i0 + ni <= nh);
#pragma omp parallel for if (ni * nj > 100000)
    for (int64_t i = i0; i < i0 + ni; i++) {
        T* __restrict simi = get_val(i);
        TI* __restrict idxi = get_ids(i);
        T* __restrict attri = get_attr(i);

        const T* ip_line = vin + (i - i0) * nj;
        const TI* id_line = id_in + (i - i0) * id_stride;
        const T* atrin_line = atrin + (i - i0) * nj;

        for (size_t j = 0; j < nj; j++) {
            T ip = ip_line[j];
            T atrin_tmp = atrin_line[j];
            if (C::cmp(simi[0], ip)) {
                heap_replace_top_one_attribute<C>(k, simi, idxi, attri, ip, id_line[j], atrin_tmp);
            }
        }
    }
}

template <typename C>
void HeapArrayOneAttribute<C>::addn_query_subset_with_ids(
        size_t nsubset,
        const TI* subset,
        size_t nj,
        const T* vin,
        const T* atrin,
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
        T* __restrict attri = get_attr(i);

        const T* ip_line = vin + si * nj;
        const TI* id_line = id_in + si * id_stride;
        const T* atrin_line = atrin + si * nj;

        for (size_t j = 0; j < nj; j++) {
            T ip = ip_line[j];
            T atrin_tmp = atrin_line[j];
            if (C::cmp(simi[0], ip)) {
                heap_replace_top_one_attribute<C>(k, simi, idxi, attri, ip, id_line[j], atrin_tmp);
            }
        }
    }
}

template <typename C>
void HeapArrayOneAttribute<C>::per_line_extrema(T* out_val, T* out_attr, TI* out_ids) const {
#pragma omp parallel for if (nh * k > 100000)
    for (int64_t j = 0; j < nh; j++) {
        int64_t imin = -1;
        typename C::T xval = C::Crev::neutral();
        typename C::T atrval = C::Crev::neutral();
        const typename C::T* x_ = val + j * k;
        const typename C::T* atr_ = attr + j * k;
        for (size_t i = 0; i < k; i++)
            if (C::cmp(x_[i], xval)) {
                xval = x_[i];
                atrval = atr_[i];
                imin = i;
            }
        if (out_val)
            out_val[j] = xval;
        if (out_attr)
            out_attr[j] = atrval;
        if (out_ids) {
            if (ids && imin != -1)
                out_ids[j] = ids[j * k + imin];
            else
                out_ids[j] = imin;
        }
    }
}

// explicit instanciations

template struct HeapArrayOneAttribute<CMin<float, int64_t>>;
template struct HeapArrayOneAttribute<CMax<float, int64_t>>;
template struct HeapArrayOneAttribute<CMin<float, int32_t>>;
template struct HeapArrayOneAttribute<CMax<float, int32_t>>;
template struct HeapArrayOneAttribute<CMin<int, int64_t>>;
template struct HeapArrayOneAttribute<CMax<int, int64_t>>;


/* For Two Attribute */
template <typename C>
void HeapArrayTwoAttribute<C>::heapify() {
#pragma omp parallel for
    for (int64_t j = 0; j < nh; j++)
        heap_heapify_two_attribute<C>(k, val + j * k, ids + j * k, attr_first + j * k, attr_second + j * k);
}

template <typename C>
void HeapArrayTwoAttribute<C>::reorder() {
#pragma omp parallel for
    for (int64_t j = 0; j < nh; j++)
        heap_reorder_two_attribute<C>(k, val + j * k, ids + j * k, attr_first + j * k, attr_second + j * k);
}

template <typename C>
void HeapArrayTwoAttribute<C>::addn(size_t nj, const T* vin, const T* atrfin, const T* atrsin, TI j0, size_t i0, int64_t ni) {
    if (ni == -1)
        ni = nh;
    assert(i0 >= 0 && i0 + ni <= nh);
#pragma omp parallel for if (ni * nj > 100000)
    for (int64_t i = i0; i < i0 + ni; i++) {
        T* __restrict simi = get_val(i);
        TI* __restrict idxi = get_ids(i);
        T* __restrict attrfi = get_attr_first(i);
        T* __restrict attrsi = get_attr_second(i);

        const T* ip_line = vin + (i - i0) * nj;
        const T* atrfin_line = atrfin + (i - i0) * nj;
        const T* atrsin_line = atrsin + (i - i0) * nj;

        for (size_t j = 0; j < nj; j++) {
            T ip = ip_line[j];
            T atrfin_tmp = atrfin_line[j];
            T atrsin_tmp = atrsin_line[j];
            if (C::cmp(simi[0], ip)) {
                heap_replace_top_two_attribute<C>(k, simi, idxi, attrfi, attrsi, ip, j + j0, atrfin_tmp, atrsin_tmp);
            }
        }
    }
}

template <typename C>
void HeapArrayTwoAttribute<C>::addn_with_ids(
        size_t nj,
        const T* vin,
        const T* atrfin,
        const T* atrsin,
        const TI* id_in,
        int64_t id_stride,
        size_t i0,
        int64_t ni) {
    if (id_in == nullptr) {
        addn(nj, vin, atrfin, atrsin, 0, i0, ni);
        return;
    }
    if (ni == -1)
        ni = nh;
    assert(i0 >= 0 && i0 + ni <= nh);
#pragma omp parallel for if (ni * nj > 100000)
    for (int64_t i = i0; i < i0 + ni; i++) {
        T* __restrict simi = get_val(i);
        TI* __restrict idxi = get_ids(i);
        T* __restrict attrfi = get_attr_first(i);
        T* __restrict attrsi = get_attr_second(i);

        const T* ip_line = vin + (i - i0) * nj;
        const TI* id_line = id_in + (i - i0) * id_stride;
        const T* atrfin_line = atrfin + (i - i0) * nj;
        const T* atrsin_line = atrsin + (i - i0) * nj;

        for (size_t j = 0; j < nj; j++) {
            T ip = ip_line[j];
            T atrfin_tmp = atrfin_line[j];
            T atrsin_tmp = atrsin_line[j];
            if (C::cmp(simi[0], ip)) {
                heap_replace_top_two_attribute<C>(k, simi, idxi, attrfi, attrsi, ip, id_line[j], atrfin_tmp, atrsin_tmp);
            }
        }
    }
}

template <typename C>
void HeapArrayTwoAttribute<C>::addn_query_subset_with_ids(
        size_t nsubset,
        const TI* subset,
        size_t nj,
        const T* vin,
        const T* atrfin,
        const T* atrsin,
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
        T* __restrict attrfi = get_attr_first(i);
        T* __restrict attrsi = get_attr_second(i);

        const T* ip_line = vin + si * nj;
        const TI* id_line = id_in + si * id_stride;
        const T* atrfin_line = atrfin + si * nj;
        const T* atrsin_line = atrsin + si * nj;

        for (size_t j = 0; j < nj; j++) {
            T ip = ip_line[j];
            T atrfin_tmp = atrfin_line[j];
            T atrsin_tmp = atrsin_line[j];
            if (C::cmp(simi[0], ip)) {
                heap_replace_top_two_attribute<C>(k, simi, idxi, attrfi, attrsi, ip, id_line[j], atrfin_tmp, atrsin_tmp);
            }
        }
    }
}

template <typename C>
void HeapArrayTwoAttribute<C>::per_line_extrema(T* out_val, T* out_attr_first, T* out_attr_second, TI* out_ids) const {
#pragma omp parallel for if (nh * k > 100000)
    for (int64_t j = 0; j < nh; j++) {
        int64_t imin = -1;
        typename C::T xval = C::Crev::neutral();
        typename C::T atrfval = C::Crev::neutral();
        typename C::T atrsval = C::Crev::neutral();
        const typename C::T* x_ = val + j * k;
        const typename C::T* atrf_ = attr_first + j * k;
        const typename C::T* atrs_ = attr_second + j * k;
        for (size_t i = 0; i < k; i++)
            if (C::cmp(x_[i], xval)) {
                xval = x_[i];
                atrfval = atrf_[i];
                atrsval = atrs_[i];
                imin = i;
            }
        if (out_val)
            out_val[j] = xval;
        if (out_attr_first)
            out_attr_first[j] = atrfval;
        if (out_attr_second)
            out_attr_second[j] = atrsval;
        if (out_ids) {
            if (ids && imin != -1)
                out_ids[j] = ids[j * k + imin];
            else
                out_ids[j] = imin;
        }
    }
}

// explicit instanciations

template struct HeapArrayTwoAttribute<CMin<float, int64_t>>;
template struct HeapArrayTwoAttribute<CMax<float, int64_t>>;
template struct HeapArrayTwoAttribute<CMin<float, int32_t>>;
template struct HeapArrayTwoAttribute<CMax<float, int32_t>>;
template struct HeapArrayTwoAttribute<CMin<int, int64_t>>;
template struct HeapArrayTwoAttribute<CMax<int, int64_t>>;

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


/* For One Attribute */
template <class idx_t, class C>
void merge_knn_results_one_attribute(
        size_t n,
        size_t k,
        typename C::TI nshard,
        const typename C::T* all_distances,
        const idx_t* all_labels,
        const typename C::T* all_attributes,
        typename C::T* distances,
        idx_t* labels,
        typename C::T* attributes) {
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
        std::vector<distance_t> buf3(nshard);
        distance_t* heap_attrs = buf3.data();
#pragma omp for
        for (long i = 0; i < n; i++) {
            // the heap maps values to the shard where they are
            // produced.
            const distance_t* D_in = all_distances + i * k;
            const idx_t* I_in = all_labels + i * k;
            const distance_t* ATR_in = all_attributes + i * k;
            int heap_size = 0;

            // push the first element of each shard (if not -1)
            for (long s = 0; s < nshard; s++) {
                pointer[s] = 0;
                if (I_in[stride * s] >= 0) {
                    heap_push_one_attribute<C>(
                            ++heap_size,
                            heap_vals,
                            shard_ids,
                            heap_attrs,
                            D_in[stride * s],
                            s,
                            ATR_in[stride * s]);
                }
            }

            distance_t* D = distances + i * k;
            idx_t* I = labels + i * k;
            distance_t* ATR = attributes + i * k;

            int j;
            for (j = 0; j < k && heap_size > 0; j++) {
                // pop element from best shard
                int s = shard_ids[0]; // top of heap
                int& p = pointer[s];
                D[j] = heap_vals[0];
                I[j] = I_in[stride * s + p];
                ATR[j] = heap_attrs[0];

                // pop from shard, advance pointer for this shard
                heap_pop_one_attribute<C>(heap_size--, heap_vals, shard_ids, heap_attrs);
                p++;
                if (p < k && I_in[stride * s + p] >= 0) {
                    heap_push_one_attribute<C>(
                            ++heap_size,
                            heap_vals,
                            shard_ids,
                            heap_attrs,
                            D_in[stride * s + p],
                            s,
                            ATR_in[stride * s + p]);
                }
            }
            for (; j < k; j++) {
                I[j] = -1;
                D[j] = C::Crev::neutral();
                ATR[j] = C::Crev::neutral();
            }
        }
    }
}

// explicit instanciations
#define INSTANTIATEONEATTRIBUTE(C, distance_t)                                  \
    template void merge_knn_results_one_attribute<int64_t, C<distance_t, int>>( \
            size_t,                                               \
            size_t,                                               \
            int,                                                  \
            const distance_t*,                                    \
            const int64_t*,                                       \
            const distance_t*,                                    \
            distance_t*,                                          \
            int64_t*,                                             \
            distance_t*);

INSTANTIATEONEATTRIBUTE(CMin, float);
INSTANTIATEONEATTRIBUTE(CMax, float);
INSTANTIATEONEATTRIBUTE(CMin, int32_t);
INSTANTIATEONEATTRIBUTE(CMax, int32_t);


/* For Two Attribute */
template <class idx_t, class C>
void merge_knn_results_two_attribute(
        size_t n,
        size_t k,
        typename C::TI nshard,
        const typename C::T* all_distances,
        const idx_t* all_labels,
        const typename C::T* all_attributes_first,
        const typename C::T* all_attributes_second,
        typename C::T* distances,
        idx_t* labels,
        typename C::T* attributes_first,
        typename C::T* attributes_second) {
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
        std::vector<distance_t> buf3(nshard);
        distance_t* heap_attrs_first = buf3.data();
        std::vector<distance_t> buf4(nshard);
        distance_t* heap_attrs_second = buf4.data();
#pragma omp for
        for (long i = 0; i < n; i++) {
            // the heap maps values to the shard where they are
            // produced.
            const distance_t* D_in = all_distances + i * k;
            const idx_t* I_in = all_labels + i * k;
            const distance_t* ATRF_in = all_attributes_first + i * k;
            const distance_t* ATRS_in = all_attributes_second + i * k;
            int heap_size = 0;

            // push the first element of each shard (if not -1)
            for (long s = 0; s < nshard; s++) {
                pointer[s] = 0;
                if (I_in[stride * s] >= 0) {
                    heap_push_two_attribute<C>(
                            ++heap_size,
                            heap_vals,
                            shard_ids,
                            heap_attrs_first,
                            heap_attrs_second,
                            D_in[stride * s],
                            s,
                            ATRF_in[stride * s],
                            ATRS_in[stride * s]);
                }
            }

            distance_t* D = distances + i * k;
            idx_t* I = labels + i * k;
            distance_t* ATRF = attributes_first + i * k;
            distance_t* ATRS = attributes_second + i * k;

            int j;
            for (j = 0; j < k && heap_size > 0; j++) {
                // pop element from best shard
                int s = shard_ids[0]; // top of heap
                int& p = pointer[s];
                D[j] = heap_vals[0];
                I[j] = I_in[stride * s + p];
                ATRF[j] = heap_attrs_first[0];
                ATRS[j] = heap_attrs_second[0];

                // pop from shard, advance pointer for this shard
                heap_pop_two_attribute<C>(heap_size--, heap_vals, shard_ids, heap_attrs_first, heap_attrs_second);
                p++;
                if (p < k && I_in[stride * s + p] >= 0) {
                    heap_push_two_attribute<C>(
                            ++heap_size,
                            heap_vals,
                            shard_ids,
                            heap_attrs_first,
                            heap_attrs_second,
                            D_in[stride * s + p],
                            s,
                            ATRF_in[stride * s + p],
                            ATRS_in[stride * s + p]);
                }
            }
            for (; j < k; j++) {
                I[j] = -1;
                D[j] = C::Crev::neutral();
                ATRF[j] = C::Crev::neutral();
                ATRS[j] = C::Crev::neutral();
            }
        }
    }
}

// explicit instanciations
#define INSTANTIATETWOATTRIBUTE(C, distance_t)                                  \
    template void merge_knn_results_two_attribute<int64_t, C<distance_t, int>>( \
            size_t,                                               \
            size_t,                                               \
            int,                                                  \
            const distance_t*,                                    \
            const int64_t*,                                       \
            const distance_t*,                                    \
            const distance_t*,                                    \
            distance_t*,                                          \
            int64_t*,                                             \
            distance_t*,                                          \
            distance_t*);

INSTANTIATETWOATTRIBUTE(CMin, float);
INSTANTIATETWOATTRIBUTE(CMax, float);
INSTANTIATETWOATTRIBUTE(CMin, int32_t);
INSTANTIATETWOATTRIBUTE(CMax, int32_t);

} // namespace faiss
