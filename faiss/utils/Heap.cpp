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
        T i = subset[si];
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
template struct HeapArray<CMin<int, int64_t>>;
template struct HeapArray<CMax<int, int64_t>>;

} // namespace faiss
