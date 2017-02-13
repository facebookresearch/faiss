
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

/* Copyright 2004-present Facebook. All Rights Reserved. */
/* Function for soft heap */

#include "Heap.h"


namespace faiss {











/* Return the scheduled capacity for soft heaps */
size_t softheap_capacity (size_t N, size_t K, size_t T, size_t * ti,
                          size_t * ki, double delta)
{
    size_t t;
    double q = (double) K / (double) N;
    size_t k0 = ceil (log (T / delta) / log (N/(K-1.0)));
    if (k0 > K)
        k0 = K;
    if (k0 <= 0) k0 = 1;

    for (t = 0; t < T; t++) {
        double n = (double) ti[t];
        double dn = delta / (double) T;
        double gamp = (N-n+1.0)/N*q + (n-1.0)/N * sqrt (2.0*q*log(2.0/dn)/n);
        double gamm = (N-n)/N * ((n+1.0) / n*q + (N-n-1.0)/n
                      * sqrt (2.0*q*log (2.0/dn) / (N-n-1.0)));
        double gam = (gamp < gamm ? gamp : gamm);
        double A = q * n;
        double B = 2.0/3.0 * log (2.0/dn);
        double C = sqrt (B * B + 2*n*log(2.0/dn)*gam);

        if (ti[t] < k0) {
            ti[t] = k0;
            ki[t] = k0;
        }

        if (ti[t] >= N) {
            ti[t] = N;
            ki[t] = K;
        }
        else {
            ki[t] = ceil (A+B+C);
            if (ki[t] > K)
                ki[t] = K;
            if (ki[t] < k0)
                ki[t] = k0;
        }
    }
    return k0;
}


/* Return the memory capacity that is theoretically required for writing
   a buffer in a radix sort, assuming that the input is shuffled */
size_t softheap_maxel (size_t N, size_t K, size_t T, double delta)
{
    double ln2T_delta = log (2.0 * T / delta);
    double lnN_K = log (N / K);
    double lnT = log (T);
    double A = K * (1.0 + (lnT + lnN_K) / T)
               + sqrt (2.0 * K * ln2T_delta)
               * (M_PI_2 + (2.0 + lnN_K) / sqrt (T))
               + 2.0 * (1.0 + lnN_K + lnT) * ln2T_delta;
    double ln2T_delta_3 = ln2T_delta / 3.0;
    double maxel = A + ln2T_delta_3
                   + sqrt (ln2T_delta_3 * ln2T_delta_3 + 2.0 * A * ln2T_delta);
    return (ceil(maxel));
}


template <typename C>
void HeapArray<C>::heapify ()
{
#pragma omp parallel for
    for (size_t j = 0; j < nh; j++)
        heap_heapify<C> (k, val + j * k, ids + j * k);
}

template <typename C>
void HeapArray<C>::reorder ()
{
#pragma omp parallel for
    for (size_t j = 0; j < nh; j++)
        heap_reorder<C> (k, val + j * k, ids + j * k);
}

template <typename C>
void HeapArray<C>::addn (size_t nj, const T *vin, TI j0,
                         size_t i0, long ni)
{
    if (ni == -1) ni = nh;
    assert (i0 >= 0 && i0 + ni <= nh);
#pragma omp parallel for
    for (size_t i = i0; i < i0 + ni; i++) {
        T * __restrict simi = get_val(i);
        TI * __restrict idxi = get_ids (i);
        const T *ip_line = vin + (i - i0) * nj;

        for (size_t j = 0; j < nj; j++) {
            T ip = ip_line [j];
            if (C::cmp(simi[0], ip)) {
                heap_pop<C> (k, simi, idxi);
                heap_push<C> (k, simi, idxi, ip, j + j0);
            }
        }
    }
}

template <typename C>
void HeapArray<C>::addn_with_ids (
     size_t nj, const T *vin, const TI *id_in,
     long id_stride, size_t i0, long ni)
{
    if (id_in == nullptr) {
        addn (nj, vin, 0, i0, ni);
        return;
    }
    if (ni == -1) ni = nh;
    assert (i0 >= 0 && i0 + ni <= nh);
#pragma omp parallel for
    for (size_t i = i0; i < i0 + ni; i++) {
        T * __restrict simi = get_val(i);
        TI * __restrict idxi = get_ids (i);
        const T *ip_line = vin + (i - i0) * nj;
        const TI *id_line = id_in + (i - i0) * id_stride;

        for (size_t j = 0; j < nj; j++) {
            T ip = ip_line [j];
            if (C::cmp(simi[0], ip)) {
                heap_pop<C> (k, simi, idxi);
                heap_push<C> (k, simi, idxi, ip, id_line [j]);
            }
        }
    }
}

template <typename C>
void HeapArray<C>::per_line_extrema (
                   T * out_val,
                   TI * out_ids) const
{
#pragma omp parallel for
    for (size_t j = 0; j < nh; j++) {
        long imin = -1;
        typename C::T xval = C::Crev::neutral ();
        const typename C::T * x_ = val + j * k;
        for (size_t i = 0; i < k; i++)
            if (C::cmp (x_[i], xval)) {
                xval = x_[i];
                imin = i;
            }
        if (out_val)
            out_val[j] = xval;

        if (out_ids) {
            if (ids && imin != -1)
                out_ids[j] = ids [j * k + imin];
            else
                out_ids[j] = imin;
        }
    }
}




// explicit instanciations

template class HeapArray<CMin <float, long> >;
template class HeapArray<CMax <float, long> >;
template class HeapArray<CMin <int, long> >;
template class HeapArray<CMax <int, long> >;





}  // END namespace fasis
