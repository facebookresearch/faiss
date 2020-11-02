/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

namespace faiss {

namespace partitioning {


template<class C>
void find_minimax(
    const typename C::T * vals, int n,
    typename C::T & smin, typename C::T & smax
) {
    smin = C::neutral();
    smax = C::Crev::neutral();

    for (size_t i = 0; i < n; i++) {
        typename C::T v = vals[i];
        if (C::cmp(smin, v)) smin = v;
        if (C::cmp(v, smax)) smax = v;
    }
}

template<class C>
void count_lt_and_eq(
    const typename C::T * vals, size_t n, typename C::T thresh,
    size_t & n_lt, size_t & n_eq
) {
    n_lt = n_eq = 0;

    for(size_t i = 0; i < n; i++) {
        typename C::T v = *vals++;
        if(C::cmp(thresh, v)) {
            n_lt++;
        } else if(v == thresh) {
            n_eq++;
        }
    }
}


template<class C>
size_t compress_array(
    typename C::T *vals, typename C::TI * ids,
    size_t n, typename C::T thresh, size_t n_eq
) {
    size_t wp = 0;
    for(size_t i = 0; i < n; i++) {
        if (C::cmp(thresh, vals[i])) {
            vals[wp] = vals[i];
            ids[wp] = ids[i];
            wp++;
        } else if (n_eq > 0 && vals[i] == thresh) {
            vals[wp] = vals[i];
            ids[wp] = ids[i];
            wp++;
            n_eq--;
        }
    }
    assert(n_eq == 0);
    return wp;
}


#define IFV if(true)

template<class C>
typename C::T partition_fuzzy_with_bounds(
    typename C::T *vals, typename C::TI * ids, size_t n,
    size_t q_min, size_t q_max, size_t * q_out,
    typename C::T s0i, typename C::T s1i)
{

    if (q_min == 0) {
        if (q_out) {
            *q_out = C::Crev::neutral();
        }
        return 0;
    }
    if (q_max >= n) {
        if (q_out) {
            *q_out = q_max;
        }
        return C::neutral();
    }
    if (s0i == s1i) {
        if (q_out) {
            *q_out = q_min;
        }
        return s0i;
    }

    using T = typename C::T;

    // lower bound inclusive, upper exclusive
    T s0 = s0i, s1 = C::nextafter(s1i);

    IFV printf("bounds: %g %g\n", s0, s1);

    T thresh;
    size_t n_eq = 0, n_lt = 0;
    size_t q = 0;

    while(s0 + 1 < s1) {
        thresh = (s0 + s1) / 2;
        count_lt_and_eq<C>(vals, n, thresh, n_lt, n_eq);

        IFV  printf("   [%g %g] thresh=%g n_lt=%ld n_eq=%ld, q=%ld:%ld/%ld\n",
            s0, s1, thresh, n_lt, n_eq, q_min, q_max, n);

        if (n_lt <= q_min) {
            if (n_lt + n_eq >= q_min) {
                q = q_min;
                break;
            } else {
                s0 = thresh;
            }
        } else if (n_lt <= q_max) {
            q = n_lt;
            break;
        } else {
            s1  = thresh;
        }
    }

    size_t n_eq_1 = q - n_lt;

    IFV printf("shrink: thresh=%g n_eq_1=%ld\n", thresh, n_eq_1);

    if (n_eq_1 < 0) { // happens when > q elements are at lower bound
        assert(s0 + 1 == s1);
        q = q_min;
        thresh = C::Crev::nextafter(thresh);
        n_eq_1 = q;
    } else {
        assert(n_eq_1 <= n_eq);
    }

    int wp = compress_array<C>(vals, ids, n, thresh, n_eq_1);

    assert(wp == q);
    if (q_out) {
        *q_out = q;
    }

    return thresh;
}


} // namespace partitioning


template<class C>
typename C::T partition_fuzzy(
    typename C::T *vals, typename C::TI * ids, size_t n,
    size_t q_min, size_t q_max, size_t * q_out)
{
    typename C::T s0i, s1i;

    partitioning::find_minimax<C>(vals, n, s0i, s1i);

    return partitioning::partition_fuzzy_with_bounds<C>(
        vals, ids, n, q_min, q_max, q_out, s0i, s1i);
}





} // namespace faiss
