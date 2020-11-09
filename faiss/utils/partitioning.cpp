/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/partitioning.h>

#include <cmath>
#include <cassert>

#include <faiss/impl/FaissAssert.h>

#include <faiss/utils/AlignedTable.h>

#include <faiss/utils/ordered_key_value.h>

#include <faiss/utils/simdlib.h>

namespace faiss {


/******************************************************************
 * Internal routines
 ******************************************************************/


namespace partitioning {

template<typename T>
T median3(T a, T b, T c) {
    if (a > b) {
        std::swap(a, b);
    }
    if (c > b) {
        return b;
    }
    if (c > a) {
        return c;
    }
    return a;
}


template<class C>
typename C::T sample_threshold_median3(
    const typename C::T * vals, int n,
    typename C::T thresh_inf, typename C::T thresh_sup
) {
    using T = typename C::T;
    size_t big_prime = 6700417;
    T val3[3];
    int vi = 0;

    for (size_t i = 0; i < n; i++) {
        T v = vals[(i * big_prime) % n];
        // thresh_inf < v < thresh_sup (for CMax)
        if (C::cmp(v, thresh_inf) && C::cmp(thresh_sup, v)) {
            val3[vi++] = v;
            if (vi == 3) {
                break;
            }
        }
    }

    if (vi == 3) {
        return median3(val3[0], val3[1], val3[2]);
    } else if (vi != 0) {
        return val3[0];
    } else {
        FAISS_THROW_MSG("too few values to compute a median");
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


#define IFV if(false)

template<class C>
typename C::T partition_fuzzy_median3(
    typename C::T *vals, typename C::TI * ids, size_t n,
    size_t q_min, size_t q_max, size_t * q_out)
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

    using T = typename C::T;

    // here we use bissection with a median of 3 to find the threshold and
    // compress the arrays afterwards. So it's a n*log(n) algoirithm rather than
    // qselect's O(n) but it avoids compressing the array.

    FAISS_THROW_IF_NOT(n >= 3);

    T thresh_inf = C::Crev::neutral();
    T thresh_sup = C::neutral();
    T thresh = median3(vals[0], vals[n / 2], vals[n - 1]);

    size_t n_eq = 0, n_lt = 0;
    size_t q = 0;

    for(int it = 0; it < 200; it++) {
        count_lt_and_eq<C>(vals, n, thresh, n_lt, n_eq);

        IFV  printf("   thresh=%g [%g %g] n_lt=%ld n_eq=%ld, q=%ld:%ld/%ld\n",
            thresh, thresh_inf, thresh_sup, n_lt, n_eq, q_min, q_max, n);

        if (n_lt <= q_min) {
            if (n_lt + n_eq >= q_min) {
                q = q_min;
                break;
            } else {
                thresh_inf = thresh;
            }
        } else if (n_lt <= q_max) {
            q = n_lt;
            break;
        } else {
            thresh_sup = thresh;
        }

        // FIXME avoid a second pass over the array to sample the threshold
        IFV  printf("     sample thresh in [%g %g]\n", thresh_inf, thresh_sup);
        thresh = sample_threshold_median3<C>(vals, n, thresh_inf, thresh_sup);
    }

    int64_t n_eq_1 = q - n_lt;

    IFV printf("shrink: thresh=%g n_eq_1=%ld\n", thresh, n_eq_1);

    if (n_eq_1 < 0) { // happens when > q elements are at lower bound
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



/******************************************************************
 * SIMD routines when vals is an aligned array of uint16_t
 ******************************************************************/



namespace simd_partitioning {



void find_minimax(
        const uint16_t * vals, size_t n,
        uint16_t & smin, uint16_t & smax
) {

    simd16uint16 vmin(0xffff), vmax(0);
    for (size_t i = 0; i + 15 < n; i += 16) {
        simd16uint16 v(vals + i);
        vmin.accu_min(v);
        vmax.accu_max(v);
    }

    uint16_t tab32[32] __attribute__ ((aligned (32)));
    vmin.store(tab32);
    vmax.store(tab32 + 16);

    smin = tab32[0], smax = tab32[16];

    for(int i = 1; i < 16; i++) {
        smin = std::min(smin, tab32[i]);
        smax = std::max(smax, tab32[i + 16]);
    }

    // missing values
    for(size_t i = (n & ~15); i < n; i++) {
        smin = std::min(smin, vals[i]);
        smax = std::max(smax, vals[i]);
    }

}


void count_lt_and_eq(
    const uint16_t * vals, int n, uint16_t thresh,
    size_t & n_lt, size_t & n_eq
) {
    n_lt = n_eq = 0;
    simd16uint16 thr16(thresh);

    size_t n1 = n / 16;

    for (size_t i = 0; i < n1; i++) {
        simd16uint16 v(vals);
        vals += 16;
        simd16uint16 eqmask = (v == thr16);
        simd16uint16 max2 = simd16uint16(_mm256_max_epu16(v.i, thr16.i));
        simd16uint16 gemask = (v == max2);
        uint32_t bits = _mm256_movemask_epi8(
            _mm256_packs_epi16(eqmask.i, gemask.i)
        );
        int i_eq = __builtin_popcount(bits & 0x00ff00ff);
        int i_ge = __builtin_popcount(bits) - i_eq;
        n_eq += i_eq;
        n_lt += 16 - i_ge;
    }

    for(size_t i = n1 * 16; i < n; i++) {
        uint16_t v = *vals++;
        if(v < thresh) {
            n_lt++;
        } else if(v == thresh) {
            n_eq++;
        }
    }
}


template<typename TI>
uint16_t simd_partition(uint16_t *vals, TI * ids, size_t n, size_t q) {

    assert(is_aligned_pointer(vals));

    if (q == 0) {
        return 0;
    }
    if (q >= n) {
        return 0xffff;
    }

    uint16_t s0i, s1i;
    find_minimax(vals, n, s0i, s1i);

    return simd_partition_fuzzy_with_bounds(
        vals, ids, n, q, q, nullptr, s0i, s1i);
}

template<typename TI>
uint16_t simd_partition_with_bounds(
    uint16_t *vals, TI * ids, size_t n, size_t q,
    uint16_t s0i, uint16_t s1i)
{
    return simd_partition_fuzzy_with_bounds(
        vals, ids, n, q, q, nullptr, s0i, s1i);
}


/* compress separated values and ids table, keeping all values < thresh and at
 * most n_eq equal values */

template<typename TI>
int simd_compress_array(
    uint16_t *vals, TI * ids, size_t n, uint16_t thresh, int n_eq
) {
    simd16uint16 thr16(thresh);
    simd16uint16 mixmask(0xff00);

    int wp = 0;
    size_t i0;

    // loop while there are eqs to collect
    for (i0 = 0; i0 + 15 < n && n_eq > 0; i0 += 16) {
        simd16uint16 v(vals + i0);
        simd16uint16 max2 = simd16uint16(_mm256_max_epu16(v.i, thr16.i));
        simd16uint16 gemask = (v == max2);
        simd16uint16 eqmask = (v == thr16);
        uint32_t bits = _mm256_movemask_epi8(
            _mm256_blendv_epi8(eqmask.i, gemask.i, mixmask.i)
        );
        bits ^= 0xAAAAAAAA;
        // bit 2*i     : eq
        // bit 2*i + 1 : lt

        while(bits) {
            int j = __builtin_ctz(bits) & (~1);
            bool is_eq = (bits >> j) & 1;
            bool is_lt = (bits >> j) & 2;
            bits &= ~(3 << j);
            j >>= 1;

            if (is_lt) {
                vals[wp] = vals[i0 + j];
                ids[wp] = ids[i0 + j];
                wp++;
            } else if(is_eq && n_eq > 0) {
                vals[wp] = vals[i0 + j];
                ids[wp] = ids[i0 + j];
                wp++;
                n_eq--;
            }
        }
    }

    // handle remaining, only striclty lt ones.
    for (; i0 + 15 < n; i0 += 16) {
        simd16uint16 v(vals + i0);
        simd16uint16 max2 = simd16uint16(_mm256_max_epu16(v.i, thr16.i));
        simd16uint16 gemask = (v == max2);
        uint32_t bits = ~_mm256_movemask_epi8(gemask.i);

        while(bits) {
            int j = __builtin_ctz(bits);
            bits &= ~(3 << j);
            j >>= 1;

            vals[wp] = vals[i0 + j];
            ids[wp] = ids[i0 + j];
            wp++;
        }
    }

    for(int i = (n & ~15); i < n; i++) {
        if (vals[i] < thresh) {
            vals[wp] = vals[i];
            ids[wp] = ids[i];
            wp++;
        } else if (vals[i] == thresh && n_eq > 0) {
            vals[wp] = vals[i];
            ids[wp] = ids[i];
            wp++;
            n_eq--;
        }
    }
    assert(n_eq == 0);
    return wp;
}




static uint64_t get_cy () {
#ifdef  MICRO_BENCHMARK
    uint32_t high, low;
    asm volatile("rdtsc \n\t"
                 : "=a" (low),
                   "=d" (high));
    return ((uint64_t)high << 32) | (low);
#else
    return 0;
#endif
}

#define IFV if(false)

template<typename TI>
uint16_t simd_partition_fuzzy_with_bounds(
    uint16_t *vals, TI * ids, size_t n,
    size_t q_min, size_t q_max, size_t * q_out,
    uint16_t s0i, uint16_t s1i)
{

    if (q_min == 0) {
        if (q_out) {
            *q_out = 0;
        }
        return 0;
    }
    if (q_max >= n) {
        if (q_out) {
            *q_out = q_max;
        }
        return 0xffff;
    }
    if (s0i == s1i) {
        if (q_out) {
            *q_out = q_min;
        }
        return s0i;
    }
    uint64_t t0 = get_cy();

    // lower bound inclusive, upper exclusive
    size_t s0 = s0i, s1 = s1i + 1;

    IFV printf("bounds: %ld %ld\n", s0, s1 - 1);

    int thresh;
    size_t n_eq = 0, n_lt = 0;
    size_t q = 0;

    while(s0 + 1 < s1) {
        thresh = (s0 + s1) / 2;
        count_lt_and_eq(vals, n, thresh, n_lt, n_eq);

        IFV  printf("   [%ld %ld] thresh=%d n_lt=%ld n_eq=%ld, q=%ld:%ld/%ld\n",
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

    uint64_t t1 = get_cy();

    int64_t n_eq_1 = q - n_lt;

    IFV printf("shrink: thresh=%d n_eq_1=%ld\n", thresh, n_eq_1);
    if (n_eq_1 < 0) { // happens when > q elements are at lower bound
        assert(s0 + 1 == s1);
        q = q_min;
        thresh--;
        n_eq_1 = q;
        IFV printf("  override: thresh=%d n_eq_1=%ld\n", thresh, n_eq_1);
    } else {
        assert(n_eq_1 <= n_eq);
    }

    size_t wp = simd_compress_array(vals, ids, n, thresh, n_eq_1);

    IFV printf("wp=%ld\n", wp);
    assert(wp == q);
    if (q_out) {
        *q_out = q;
    }
/*
    QSelect_stats.t1 += t1 - t0;
    QSelect_stats.t2 += get_cy() - t1;
*/
    return thresh;
}


template<typename TI>
uint16_t simd_partition_fuzzy(
    uint16_t *vals, TI * ids, size_t n,
    size_t q_min, size_t q_max, size_t * q_out
) {

    assert(is_aligned_pointer(vals, 32));

    uint64_t t0 = get_cy();
    uint16_t s0i, s1i;
    find_minimax(vals, n, s0i, s1i);
    // QSelect_stats.t0 += get_cy() - t0;

    return simd_partition_fuzzy_with_bounds(
        vals, ids, n, q_min, q_max, q_out, s0i, s1i);
}


} // namespace simd_partitioning


/******************************************************************
 * Driver routine
 ******************************************************************/


template<class C>
typename C::T partition_fuzzy(
    typename C::T *vals, typename C::TI * ids, size_t n,
    size_t q_min, size_t q_max, size_t * q_out)
{
#ifdef __AVX2__
    constexpr bool is_uint16 = std::is_same<typename C::T, uint16_t>::value;
    if (is_uint16 && is_aligned_pointer(vals)) {
        return simd_partitioning::simd_partition_fuzzy(
            (uint16_t*)vals, ids, n, q_min, q_max, q_out);
    }
#endif
    return partitioning::partition_fuzzy_median3<C>(
        vals, ids, n, q_min, q_max, q_out);
}


// explicit template instanciations

template float partition_fuzzy<CMin<float, int64_t>> (
    float *vals, int64_t * ids, size_t n,
    size_t q_min, size_t q_max, size_t * q_out);

template float partition_fuzzy<CMax<float, int64_t>> (
    float *vals, int64_t * ids, size_t n,
    size_t q_min, size_t q_max, size_t * q_out);

template uint16_t partition_fuzzy<CMin<uint16_t, int64_t>> (
    uint16_t *vals, int64_t * ids, size_t n,
    size_t q_min, size_t q_max, size_t * q_out);

template uint16_t partition_fuzzy<CMax<uint16_t, int64_t>> (
    uint16_t *vals, int64_t * ids, size_t n,
    size_t q_min, size_t q_max, size_t * q_out);



} // namespace faiss
