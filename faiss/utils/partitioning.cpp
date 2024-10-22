/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/partitioning.h>

#include <cassert>
#include <cmath>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/ordered_key_value.h>
#include <faiss/utils/simdlib.h>

#include <faiss/impl/platform_macros.h>

namespace faiss {

/******************************************************************
 * Internal routines
 ******************************************************************/

namespace partitioning {

template <typename T>
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

template <class C>
typename C::T sample_threshold_median3(
        const typename C::T* vals,
        int n,
        typename C::T thresh_inf,
        typename C::T thresh_sup) {
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
        return thresh_inf;
        //   FAISS_THROW_MSG("too few values to compute a median");
    }
}

template <class C>
void count_lt_and_eq(
        const typename C::T* vals,
        size_t n,
        typename C::T thresh,
        size_t& n_lt,
        size_t& n_eq) {
    n_lt = n_eq = 0;

    for (size_t i = 0; i < n; i++) {
        typename C::T v = *vals++;
        if (C::cmp(thresh, v)) {
            n_lt++;
        } else if (v == thresh) {
            n_eq++;
        }
    }
}

template <class C>
size_t compress_array(
        typename C::T* vals,
        typename C::TI* ids,
        size_t n,
        typename C::T thresh,
        size_t n_eq) {
    size_t wp = 0;
    for (size_t i = 0; i < n; i++) {
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

#define IFV if (false)

template <class C>
typename C::T partition_fuzzy_median3(
        typename C::T* vals,
        typename C::TI* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out) {
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
    // qselect's O(n) but it avoids shuffling around the array.

    FAISS_THROW_IF_NOT(n >= 3);

    T thresh_inf = C::Crev::neutral();
    T thresh_sup = C::neutral();
    T thresh = median3(vals[0], vals[n / 2], vals[n - 1]);

    size_t n_eq = 0, n_lt = 0;
    size_t q = 0;

    for (int it = 0; it < 200; it++) {
        count_lt_and_eq<C>(vals, n, thresh, n_lt, n_eq);

        IFV printf(
                "   thresh=%g [%g %g] n_lt=%ld n_eq=%ld, q=%ld:%ld/%ld\n",
                float(thresh),
                float(thresh_inf),
                float(thresh_sup),
                long(n_lt),
                long(n_eq),
                long(q_min),
                long(q_max),
                long(n));

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
        IFV printf(
                "     sample thresh in [%g %g]\n",
                float(thresh_inf),
                float(thresh_sup));
        T new_thresh =
                sample_threshold_median3<C>(vals, n, thresh_inf, thresh_sup);
        if (new_thresh == thresh_inf) {
            // then there is nothing between thresh_inf and thresh_sup
            break;
        }
        thresh = new_thresh;
    }

    int64_t n_eq_1 = q - n_lt;

    IFV printf("shrink: thresh=%g n_eq_1=%ld\n", float(thresh), long(n_eq_1));

    if (n_eq_1 < 0) { // happens when > q elements are at lower bound
        q = q_min;
        thresh = C::Crev::nextafter(thresh);
        n_eq_1 = q;
    } else {
        assert(n_eq_1 <= n_eq);
    }

    [[maybe_unused]] const int wp =
            compress_array<C>(vals, ids, n, thresh, n_eq_1);

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
        const uint16_t* vals,
        size_t n,
        uint16_t& smin,
        uint16_t& smax) {
    simd16uint16 vmin(0xffff), vmax(0);
    for (size_t i = 0; i + 15 < n; i += 16) {
        simd16uint16 v(vals + i);
        vmin.accu_min(v);
        vmax.accu_max(v);
    }

    ALIGNED(32) uint16_t tab32[32];
    vmin.store(tab32);
    vmax.store(tab32 + 16);

    smin = tab32[0], smax = tab32[16];

    for (int i = 1; i < 16; i++) {
        smin = std::min(smin, tab32[i]);
        smax = std::max(smax, tab32[i + 16]);
    }

    // missing values
    for (size_t i = (n & ~15); i < n; i++) {
        smin = std::min(smin, vals[i]);
        smax = std::max(smax, vals[i]);
    }
}

// max func differentiates between CMin and CMax (keep lowest or largest)
template <class C>
simd16uint16 max_func(simd16uint16 v, simd16uint16 thr16) {
    constexpr bool is_max = C::is_max;
    if (is_max) {
        return max(v, thr16);
    } else {
        return min(v, thr16);
    }
}

template <class C>
void count_lt_and_eq(
        const uint16_t* vals,
        int n,
        uint16_t thresh,
        size_t& n_lt,
        size_t& n_eq) {
    n_lt = n_eq = 0;
    simd16uint16 thr16(thresh);

    size_t n1 = n / 16;

    for (size_t i = 0; i < n1; i++) {
        simd16uint16 v(vals);
        vals += 16;
        simd16uint16 eqmask = (v == thr16);
        simd16uint16 max2 = max_func<C>(v, thr16);
        simd16uint16 gemask = (v == max2);
        uint32_t bits = get_MSBs(uint16_to_uint8_saturate(eqmask, gemask));
        int i_eq = __builtin_popcount(bits & 0x00ff00ff);
        int i_ge = __builtin_popcount(bits) - i_eq;
        n_eq += i_eq;
        n_lt += 16 - i_ge;
    }

    for (size_t i = n1 * 16; i < n; i++) {
        uint16_t v = *vals++;
        if (C::cmp(thresh, v)) {
            n_lt++;
        } else if (v == thresh) {
            n_eq++;
        }
    }
}

/* compress separated values and ids table, keeping all values < thresh and at
 * most n_eq equal values */
template <class C>
int simd_compress_array(
        uint16_t* vals,
        typename C::TI* ids,
        size_t n,
        uint16_t thresh,
        int n_eq) {
    simd16uint16 thr16(thresh);
    simd16uint16 mixmask(0xff00);

    int wp = 0;
    size_t i0;

    // loop while there are eqs to collect
    for (i0 = 0; i0 + 15 < n && n_eq > 0; i0 += 16) {
        simd16uint16 v(vals + i0);
        simd16uint16 max2 = max_func<C>(v, thr16);
        simd16uint16 gemask = (v == max2);
        simd16uint16 eqmask = (v == thr16);
        uint32_t bits = get_MSBs(
                blendv(simd32uint8(eqmask),
                       simd32uint8(gemask),
                       simd32uint8(mixmask)));
        bits ^= 0xAAAAAAAA;
        // bit 2*i     : eq
        // bit 2*i + 1 : lt

        while (bits) {
            int j = __builtin_ctz(bits) & (~1);
            bool is_eq = (bits >> j) & 1;
            bool is_lt = (bits >> j) & 2;
            bits &= ~(3 << j);
            j >>= 1;

            if (is_lt) {
                vals[wp] = vals[i0 + j];
                ids[wp] = ids[i0 + j];
                wp++;
            } else if (is_eq && n_eq > 0) {
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
        simd16uint16 max2 = max_func<C>(v, thr16);
        simd16uint16 gemask = (v == max2);
        uint32_t bits = ~get_MSBs(simd32uint8(gemask));

        while (bits) {
            int j = __builtin_ctz(bits);
            bits &= ~(3 << j);
            j >>= 1;

            vals[wp] = vals[i0 + j];
            ids[wp] = ids[i0 + j];
            wp++;
        }
    }

    // end with scalar
    for (int i = (n & ~15); i < n; i++) {
        if (C::cmp(thresh, vals[i])) {
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

// #define MICRO_BENCHMARK

static uint64_t get_cy() {
#ifdef MICRO_BENCHMARK
    uint32_t high, low;
    asm volatile("rdtsc \n\t" : "=a"(low), "=d"(high));
    return ((uint64_t)high << 32) | (low);
#else
    return 0;
#endif
}

#define IFV if (false)

template <class C>
uint16_t simd_partition_fuzzy_with_bounds(
        uint16_t* vals,
        typename C::TI* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out,
        uint16_t s0i,
        uint16_t s1i) {
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

    for (int it = 0; it < 200; it++) {
        // while(s0 + 1 < s1) {
        thresh = (s0 + s1) / 2;
        count_lt_and_eq<C>(vals, n, thresh, n_lt, n_eq);

        IFV printf(
                "   [%ld %ld] thresh=%d n_lt=%ld n_eq=%ld, q=%ld:%ld/%ld\n",
                s0,
                s1,
                thresh,
                n_lt,
                n_eq,
                q_min,
                q_max,
                n);
        if (n_lt <= q_min) {
            if (n_lt + n_eq >= q_min) {
                q = q_min;
                break;
            } else {
                if (C::is_max) {
                    s0 = thresh;
                } else {
                    s1 = thresh;
                }
            }
        } else if (n_lt <= q_max) {
            q = n_lt;
            break;
        } else {
            if (C::is_max) {
                s1 = thresh;
            } else {
                s0 = thresh;
            }
        }
    }

    uint64_t t1 = get_cy();

    // number of equal values to keep
    int64_t n_eq_1 = q - n_lt;

    IFV printf("shrink: thresh=%d q=%ld n_eq_1=%ld\n", thresh, q, n_eq_1);
    if (n_eq_1 < 0) { // happens when > q elements are at lower bound
        assert(s0 + 1 == s1);
        q = q_min;
        if (C::is_max) {
            thresh--;
        } else {
            thresh++;
        }
        n_eq_1 = q;
        IFV printf("  override: thresh=%d n_eq_1=%ld\n", thresh, n_eq_1);
    } else {
        assert(n_eq_1 <= n_eq);
    }

    size_t wp = simd_compress_array<C>(vals, ids, n, thresh, n_eq_1);

    IFV printf("wp=%ld\n", wp);
    assert(wp == q);
    if (q_out) {
        *q_out = q;
    }

    uint64_t t2 = get_cy();

    partition_stats.bissect_cycles += t1 - t0;
    partition_stats.compress_cycles += t2 - t1;

    return thresh;
}

template <class C>
uint16_t simd_partition_fuzzy_with_bounds_histogram(
        uint16_t* vals,
        typename C::TI* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out,
        uint16_t s0i,
        uint16_t s1i) {
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

    IFV printf(
            "partition fuzzy, q=%ld:%ld / %ld, bounds=%d %d\n",
            q_min,
            q_max,
            n,
            s0i,
            s1i);

    if (!C::is_max) {
        IFV printf(
                "revert due to CMin, q_min:q_max -> %ld:%ld\n", q_min, q_max);
        q_min = n - q_min;
        q_max = n - q_max;
    }

    // lower and upper bound of range, inclusive
    int s0 = s0i, s1 = s1i;
    // number of values < s0 and > s1
    size_t n_lt = 0, n_gt = 0;

    // output of loop:
    int thresh;          // final threshold
    uint64_t tot_eq = 0; // total nb of equal values
    uint64_t n_eq = 0;   // nb of equal values to keep
    size_t q;            // final quantile

    // buffer for the histograms
    int hist[16];

    for (int it = 0; it < 20; it++) {
        // otherwise we would be done already

        int shift = 0;

        IFV printf(
                "  it %d bounds: %d %d n_lt=%ld n_gt=%ld\n",
                it,
                s0,
                s1,
                n_lt,
                n_gt);

        int maxval = s1 - s0;

        while (maxval > 15) {
            shift++;
            maxval >>= 1;
        }

        IFV printf(
                "    histogram shift %d maxval %d ?= %d\n",
                shift,
                maxval,
                int((s1 - s0) >> shift));

        if (maxval > 7) {
            simd_histogram_16(vals, n, s0, shift, hist);
        } else {
            simd_histogram_8(vals, n, s0, shift, hist);
        }
        IFV {
            int sum = n_lt + n_gt;
            printf("    n_lt=%ld hist=[", n_lt);
            for (int i = 0; i <= maxval; i++) {
                printf("%d ", hist[i]);
                sum += hist[i];
            }
            printf("] n_gt=%ld sum=%d\n", n_gt, sum);
            assert(sum == n);
        }

        size_t sum_below = n_lt;
        int i;
        for (i = 0; i <= maxval; i++) {
            sum_below += hist[i];
            if (sum_below >= q_min) {
                break;
            }
        }
        IFV printf("    i=%d sum_below=%ld\n", i, sum_below);
        if (i <= maxval) {
            s0 = s0 + (i << shift);
            s1 = s0 + (1 << shift) - 1;
            n_lt = sum_below - hist[i];
            n_gt = n - sum_below;
        } else {
            assert(!"not implemented");
        }

        IFV printf(
                "    new bin: s0=%d s1=%d n_lt=%ld n_gt=%ld\n",
                s0,
                s1,
                n_lt,
                n_gt);

        if (s1 > s0) {
            if (n_lt >= q_min && q_max >= n_lt) {
                IFV printf("    FOUND1\n");
                thresh = s0;
                q = n_lt;
                break;
            }

            size_t n_lt_2 = n - n_gt;
            if (n_lt_2 >= q_min && q_max >= n_lt_2) {
                thresh = s1 + 1;
                q = n_lt_2;
                IFV printf("    FOUND2\n");
                break;
            }
        } else {
            thresh = s0;
            q = q_min;
            tot_eq = n - n_gt - n_lt;
            n_eq = q_min - n_lt;
            IFV printf("    FOUND3\n");
            break;
        }
    }

    IFV printf("end bissection: thresh=%d q=%ld n_eq=%ld\n", thresh, q, n_eq);

    if (!C::is_max) {
        if (n_eq == 0) {
            thresh--;
        } else {
            // thresh unchanged
            n_eq = tot_eq - n_eq;
        }
        q = n - q;
        IFV printf("revert due to CMin, q->%ld n_eq->%ld\n", q, n_eq);
    }

    size_t wp = simd_compress_array<C>(vals, ids, n, thresh, n_eq);
    IFV printf("wp=%ld ?= %ld\n", wp, q);
    assert(wp == q);
    if (q_out) {
        *q_out = wp;
    }

    return thresh;
}

template <class C>
uint16_t simd_partition_fuzzy(
        uint16_t* vals,
        typename C::TI* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out) {
    assert(is_aligned_pointer(vals));

    uint16_t s0i, s1i;
    find_minimax(vals, n, s0i, s1i);
    // QSelect_stats.t0 += get_cy() - t0;

    return simd_partition_fuzzy_with_bounds<C>(
            vals, ids, n, q_min, q_max, q_out, s0i, s1i);
}

template <class C>
uint16_t simd_partition(
        uint16_t* vals,
        typename C::TI* ids,
        size_t n,
        size_t q) {
    assert(is_aligned_pointer(vals));

    if (q == 0) {
        return 0;
    }
    if (q >= n) {
        return 0xffff;
    }

    uint16_t s0i, s1i;
    find_minimax(vals, n, s0i, s1i);

    return simd_partition_fuzzy_with_bounds<C>(
            vals, ids, n, q, q, nullptr, s0i, s1i);
}

template <class C>
uint16_t simd_partition_with_bounds(
        uint16_t* vals,
        typename C::TI* ids,
        size_t n,
        size_t q,
        uint16_t s0i,
        uint16_t s1i) {
    return simd_partition_fuzzy_with_bounds<C>(
            vals, ids, n, q, q, nullptr, s0i, s1i);
}

} // namespace simd_partitioning

/******************************************************************
 * Driver routine
 ******************************************************************/

template <class C>
typename C::T partition_fuzzy(
        typename C::T* vals,
        typename C::TI* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out) {
#ifdef __AVX2__
    constexpr bool is_uint16 = std::is_same<typename C::T, uint16_t>::value;
    if (is_uint16 && is_aligned_pointer(vals)) {
        return simd_partitioning::simd_partition_fuzzy<C>(
                (uint16_t*)vals, ids, n, q_min, q_max, q_out);
    }
#endif
    return partitioning::partition_fuzzy_median3<C>(
            vals, ids, n, q_min, q_max, q_out);
}

// explicit template instanciations

template float partition_fuzzy<CMin<float, int64_t>>(
        float* vals,
        int64_t* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out);

template float partition_fuzzy<CMax<float, int64_t>>(
        float* vals,
        int64_t* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out);

template uint16_t partition_fuzzy<CMin<uint16_t, int64_t>>(
        uint16_t* vals,
        int64_t* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out);

template uint16_t partition_fuzzy<CMax<uint16_t, int64_t>>(
        uint16_t* vals,
        int64_t* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out);

template uint16_t partition_fuzzy<CMin<uint16_t, int>>(
        uint16_t* vals,
        int* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out);

template uint16_t partition_fuzzy<CMax<uint16_t, int>>(
        uint16_t* vals,
        int* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out);

/******************************************************************
 * Histogram subroutines
 ******************************************************************/

#if defined(__AVX2__) || defined(__aarch64__)
/// FIXME when MSB of uint16 is set
// this code does not compile properly with GCC 7.4.0

namespace {

/************************************************************
 * 8 bins
 ************************************************************/

simd32uint8 accu4to8(simd16uint16 a4) {
    simd16uint16 mask4(0x0f0f);

    simd16uint16 a8_0 = a4 & mask4;
    simd16uint16 a8_1 = (a4 >> 4) & mask4;

    return simd32uint8(hadd(a8_0, a8_1));
}

simd16uint16 accu8to16(simd32uint8 a8) {
    simd16uint16 mask8(0x00ff);

    simd16uint16 a8_0 = simd16uint16(a8) & mask8;
    simd16uint16 a8_1 = (simd16uint16(a8) >> 8) & mask8;

    return hadd(a8_0, a8_1);
}

static const simd32uint8 shifts = simd32uint8::create<
        1,
        16,
        0,
        0,
        4,
        64,
        0,
        0,
        0,
        0,
        1,
        16,
        0,
        0,
        4,
        64,
        1,
        16,
        0,
        0,
        4,
        64,
        0,
        0,
        0,
        0,
        1,
        16,
        0,
        0,
        4,
        64>();

// 2-bit accumulator: we can add only up to 3 elements
// on output we return 2*4-bit results
// preproc returns either an index in 0..7 or 0xffff
// that yields a 0 when used in the table look-up
template <int N, class Preproc>
void compute_accu2(
        const uint16_t*& data,
        Preproc& pp,
        simd16uint16& a4lo,
        simd16uint16& a4hi) {
    simd16uint16 mask2(0x3333);
    simd16uint16 a2((uint16_t)0); // 2-bit accu
    for (int j = 0; j < N; j++) {
        simd16uint16 v(data);
        data += 16;
        v = pp(v);
        // 0x800 -> force second half of table
        simd16uint16 idx = v | (v << 8) | simd16uint16(0x800);
        a2 += simd16uint16(shifts.lookup_2_lanes(simd32uint8(idx)));
    }
    a4lo += a2 & mask2;
    a4hi += (a2 >> 2) & mask2;
}

template <class Preproc>
simd16uint16 histogram_8(const uint16_t* data, Preproc pp, size_t n_in) {
    assert(n_in % 16 == 0);
    int n = n_in / 16;

    simd32uint8 a8lo(0);
    simd32uint8 a8hi(0);

    for (int i0 = 0; i0 < n; i0 += 15) {
        simd16uint16 a4lo(0); // 4-bit accus
        simd16uint16 a4hi(0);

        int i1 = std::min(i0 + 15, n);
        int i;
        for (i = i0; i + 2 < i1; i += 3) {
            compute_accu2<3>(data, pp, a4lo, a4hi); // adds 3 max
        }
        switch (i1 - i) {
            case 2:
                compute_accu2<2>(data, pp, a4lo, a4hi);
                break;
            case 1:
                compute_accu2<1>(data, pp, a4lo, a4hi);
                break;
        }

        a8lo += accu4to8(a4lo);
        a8hi += accu4to8(a4hi);
    }

    // move to 16-bit accu
    simd16uint16 a16lo = accu8to16(a8lo);
    simd16uint16 a16hi = accu8to16(a8hi);

    simd16uint16 a16 = hadd(a16lo, a16hi);

    // the 2 lanes must still be combined
    return a16;
}

/************************************************************
 * 16 bins
 ************************************************************/

static const simd32uint8 shifts2 = simd32uint8::create<
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128>();

simd32uint8 shiftr_16(simd32uint8 x, int n) {
    return simd32uint8(simd16uint16(x) >> n);
}

// 2-bit accumulator: we can add only up to 3 elements
// on output we return 2*4-bit results
template <int N, class Preproc>
void compute_accu2_16(
        const uint16_t*& data,
        Preproc pp,
        simd32uint8& a4_0,
        simd32uint8& a4_1,
        simd32uint8& a4_2,
        simd32uint8& a4_3) {
    simd32uint8 mask1(0x55);
    simd32uint8 a2_0; // 2-bit accu
    simd32uint8 a2_1; // 2-bit accu
    a2_0.clear();
    a2_1.clear();

    for (int j = 0; j < N; j++) {
        simd16uint16 v(data);
        data += 16;
        v = pp(v);

        simd16uint16 idx = v | (v << 8);
        simd32uint8 a1 = shifts2.lookup_2_lanes(simd32uint8(idx));
        // contains 0s for out-of-bounds elements

        simd16uint16 lt8 = (v >> 3) == simd16uint16(0);
        lt8 = lt8 ^ simd16uint16(0xff00);

        a1 = a1 & lt8;

        a2_0 += a1 & mask1;
        a2_1 += shiftr_16(a1, 1) & mask1;
    }
    simd32uint8 mask2(0x33);

    a4_0 += a2_0 & mask2;
    a4_1 += a2_1 & mask2;
    a4_2 += shiftr_16(a2_0, 2) & mask2;
    a4_3 += shiftr_16(a2_1, 2) & mask2;
}

simd32uint8 accu4to8_2(simd32uint8 a4_0, simd32uint8 a4_1) {
    simd32uint8 mask4(0x0f);

    simd16uint16 a8_0 = combine2x2(
            (simd16uint16)(a4_0 & mask4),
            (simd16uint16)(shiftr_16(a4_0, 4) & mask4));

    simd16uint16 a8_1 = combine2x2(
            (simd16uint16)(a4_1 & mask4),
            (simd16uint16)(shiftr_16(a4_1, 4) & mask4));

    return simd32uint8(hadd(a8_0, a8_1));
}

template <class Preproc>
simd16uint16 histogram_16(const uint16_t* data, Preproc pp, size_t n_in) {
    assert(n_in % 16 == 0);
    int n = n_in / 16;

    simd32uint8 a8lo((uint8_t)0);
    simd32uint8 a8hi((uint8_t)0);

    for (int i0 = 0; i0 < n; i0 += 7) {
        simd32uint8 a4_0(0); // 0, 4, 8, 12
        simd32uint8 a4_1(0); // 1, 5, 9, 13
        simd32uint8 a4_2(0); // 2, 6, 10, 14
        simd32uint8 a4_3(0); // 3, 7, 11, 15

        int i1 = std::min(i0 + 7, n);
        int i;
        for (i = i0; i + 2 < i1; i += 3) {
            compute_accu2_16<3>(data, pp, a4_0, a4_1, a4_2, a4_3);
        }
        switch (i1 - i) {
            case 2:
                compute_accu2_16<2>(data, pp, a4_0, a4_1, a4_2, a4_3);
                break;
            case 1:
                compute_accu2_16<1>(data, pp, a4_0, a4_1, a4_2, a4_3);
                break;
        }

        a8lo += accu4to8_2(a4_0, a4_1);
        a8hi += accu4to8_2(a4_2, a4_3);
    }

    // move to 16-bit accu
    simd16uint16 a16lo = accu8to16(a8lo);
    simd16uint16 a16hi = accu8to16(a8hi);

    simd16uint16 a16 = hadd(a16lo, a16hi);

    a16 = simd16uint16{simd8uint32{a16}.unzip()};

    return a16;
}

struct PreprocNOP {
    simd16uint16 operator()(simd16uint16 x) {
        return x;
    }
};

template <int shift, int nbin>
struct PreprocMinShift {
    simd16uint16 min16;
    simd16uint16 max16;

    explicit PreprocMinShift(uint16_t min) {
        min16.set1(min);
        int vmax0 = std::min((nbin << shift) + min, 65536);
        uint16_t vmax = uint16_t(vmax0 - 1 - min);
        max16.set1(vmax); // vmax inclusive
    }

    simd16uint16 operator()(simd16uint16 x) {
        x = x - min16;
        simd16uint16 mask = (x == max(x, max16)) - (x == max16);
        return (x >> shift) | mask;
    }
};

/* unbounded versions of the functions */

void simd_histogram_8_unbounded(const uint16_t* data, int n, int* hist) {
    PreprocNOP pp;
    simd16uint16 a16 = histogram_8(data, pp, (n & ~15));

    ALIGNED(32) uint16_t a16_tab[16];
    a16.store(a16_tab);

    for (int i = 0; i < 8; i++) {
        hist[i] = a16_tab[i] + a16_tab[i + 8];
    }

    for (int i = (n & ~15); i < n; i++) {
        hist[data[i]]++;
    }
}

void simd_histogram_16_unbounded(const uint16_t* data, int n, int* hist) {
    simd16uint16 a16 = histogram_16(data, PreprocNOP(), (n & ~15));

    ALIGNED(32) uint16_t a16_tab[16];
    a16.store(a16_tab);

    for (int i = 0; i < 16; i++) {
        hist[i] = a16_tab[i];
    }

    for (int i = (n & ~15); i < n; i++) {
        hist[data[i]]++;
    }
}

} // anonymous namespace

/************************************************************
 * Driver routines
 ************************************************************/

void simd_histogram_8(
        const uint16_t* data,
        int n,
        uint16_t min,
        int shift,
        int* hist) {
    if (shift < 0) {
        simd_histogram_8_unbounded(data, n, hist);
        return;
    }

    simd16uint16 a16;

#define DISPATCH(s)                                                     \
    case s:                                                             \
        a16 = histogram_8(data, PreprocMinShift<s, 8>(min), (n & ~15)); \
        break

    switch (shift) {
        DISPATCH(0);
        DISPATCH(1);
        DISPATCH(2);
        DISPATCH(3);
        DISPATCH(4);
        DISPATCH(5);
        DISPATCH(6);
        DISPATCH(7);
        DISPATCH(8);
        DISPATCH(9);
        DISPATCH(10);
        DISPATCH(11);
        DISPATCH(12);
        DISPATCH(13);
        default:
            FAISS_THROW_FMT("dispatch for shift=%d not instantiated", shift);
    }
#undef DISPATCH

    ALIGNED(32) uint16_t a16_tab[16];
    a16.store(a16_tab);

    for (int i = 0; i < 8; i++) {
        hist[i] = a16_tab[i] + a16_tab[i + 8];
    }

    // complete with remaining bins
    for (int i = (n & ~15); i < n; i++) {
        if (data[i] < min)
            continue;
        uint16_t v = data[i] - min;
        v >>= shift;
        if (v < 8)
            hist[v]++;
    }
}

void simd_histogram_16(
        const uint16_t* data,
        int n,
        uint16_t min,
        int shift,
        int* hist) {
    if (shift < 0) {
        simd_histogram_16_unbounded(data, n, hist);
        return;
    }

    simd16uint16 a16;

#define DISPATCH(s)                                                       \
    case s:                                                               \
        a16 = histogram_16(data, PreprocMinShift<s, 16>(min), (n & ~15)); \
        break

    switch (shift) {
        DISPATCH(0);
        DISPATCH(1);
        DISPATCH(2);
        DISPATCH(3);
        DISPATCH(4);
        DISPATCH(5);
        DISPATCH(6);
        DISPATCH(7);
        DISPATCH(8);
        DISPATCH(9);
        DISPATCH(10);
        DISPATCH(11);
        DISPATCH(12);
        default:
            FAISS_THROW_FMT("dispatch for shift=%d not instantiated", shift);
    }
#undef DISPATCH

    ALIGNED(32) uint16_t a16_tab[16];
    a16.store(a16_tab);

    for (int i = 0; i < 16; i++) {
        hist[i] = a16_tab[i];
    }

    for (int i = (n & ~15); i < n; i++) {
        if (data[i] < min)
            continue;
        uint16_t v = data[i] - min;
        v >>= shift;
        if (v < 16)
            hist[v]++;
    }
}

// no AVX2
#else

void simd_histogram_16(
        const uint16_t* data,
        int n,
        uint16_t min,
        int shift,
        int* hist) {
    memset(hist, 0, sizeof(*hist) * 16);
    if (shift < 0) {
        for (size_t i = 0; i < n; i++) {
            hist[data[i]]++;
        }
    } else {
        int vmax0 = std::min((16 << shift) + min, 65536);
        uint16_t vmax = uint16_t(vmax0 - 1 - min);

        for (size_t i = 0; i < n; i++) {
            uint16_t v = data[i];
            v -= min;
            if (!(v <= vmax))
                continue;
            v >>= shift;
            hist[v]++;

            /*
            if (data[i] < min) continue;
            uint16_t v = data[i] - min;
            v >>= shift;
            if (v < 16) hist[v]++;
            */
        }
    }
}

void simd_histogram_8(
        const uint16_t* data,
        int n,
        uint16_t min,
        int shift,
        int* hist) {
    memset(hist, 0, sizeof(*hist) * 8);
    if (shift < 0) {
        for (size_t i = 0; i < n; i++) {
            hist[data[i]]++;
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            if (data[i] < min)
                continue;
            uint16_t v = data[i] - min;
            v >>= shift;
            if (v < 8)
                hist[v]++;
        }
    }
}

#endif

void PartitionStats::reset() {
    memset(this, 0, sizeof(*this));
}

PartitionStats partition_stats;

} // namespace faiss
