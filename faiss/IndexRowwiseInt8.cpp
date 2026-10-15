/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexRowwiseInt8.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#define FAISS_ROWWISE_INT8_VNNI 1
#endif

namespace faiss {

namespace {

inline const int8_t* code_ptr(const uint8_t* code) {
    return reinterpret_cast<const int8_t*>(code);
}

inline float code_scale(const uint8_t* code, size_t d) {
    float s;
    memcpy(&s, code + d, sizeof(float));
    return s;
}

int32_t dot_i8(const int8_t* a, const int8_t* b, size_t d) {
    int32_t s = 0;
    for (size_t j = 0; j < d; j++) {
        s += int32_t(a[j]) * int32_t(b[j]);
    }
    return s;
}

int32_t sum_i8(const int8_t* a, size_t d) {
    int32_t s = 0;
    for (size_t j = 0; j < d; j++) {
        s += a[j];
    }
    return s;
}

#ifdef FAISS_ROWWISE_INT8_VNNI

bool cpu_has_vnni() {
    static const bool has = __builtin_cpu_supports("avx512f") &&
            __builtin_cpu_supports("avx512bw") &&
            __builtin_cpu_supports("avx512vnni");
    return has;
}

/* out[i] = qu_i . b for nq queries (rows of qu, d bytes each) and one bank
 * vector b. qu = c_q + 128 (unsigned), so qu . b = c_q . b + 128 sum(b);
 * the caller removes the 128 sum(b) term. vpdpbusd adds groups of four
 * u8 * s8 products into int32 without saturation: the sums are exact.
 * d must be a multiple of 64. */
__attribute__((target("avx512f,avx512bw,avx512vnni"))) void dots_vnni(
        const uint8_t* qu,
        size_t nq,
        const int8_t* b,
        size_t d,
        int32_t* out) {
    size_t i = 0;
    for (; i + 4 <= nq; i += 4) {
        const uint8_t* q0 = qu + i * d;
        const uint8_t* q1 = q0 + d;
        const uint8_t* q2 = q1 + d;
        const uint8_t* q3 = q2 + d;
        __m512i a0 = _mm512_setzero_si512();
        __m512i a1 = _mm512_setzero_si512();
        __m512i a2 = _mm512_setzero_si512();
        __m512i a3 = _mm512_setzero_si512();
        for (size_t j = 0; j < d; j += 64) {
            __m512i bv = _mm512_loadu_si512(b + j);
            a0 = _mm512_dpbusd_epi32(a0, _mm512_loadu_si512(q0 + j), bv);
            a1 = _mm512_dpbusd_epi32(a1, _mm512_loadu_si512(q1 + j), bv);
            a2 = _mm512_dpbusd_epi32(a2, _mm512_loadu_si512(q2 + j), bv);
            a3 = _mm512_dpbusd_epi32(a3, _mm512_loadu_si512(q3 + j), bv);
        }
        out[i] = _mm512_reduce_add_epi32(a0);
        out[i + 1] = _mm512_reduce_add_epi32(a1);
        out[i + 2] = _mm512_reduce_add_epi32(a2);
        out[i + 3] = _mm512_reduce_add_epi32(a3);
    }
    for (; i < nq; i++) {
        const uint8_t* q0 = qu + i * d;
        __m512i a0 = _mm512_setzero_si512();
        for (size_t j = 0; j < d; j += 64) {
            a0 = _mm512_dpbusd_epi32(
                    a0, _mm512_loadu_si512(q0 + j), _mm512_loadu_si512(b + j));
        }
        out[i] = _mm512_reduce_add_epi32(a0);
    }
}

#else

bool cpu_has_vnni() {
    return false;
}

void dots_vnni(const uint8_t*, size_t, const int8_t*, size_t, int32_t*) {
    FAISS_THROW_MSG("VNNI kernel not compiled");
}

#endif

/* exhaustive search over the codes, one block of queries per task.
 * C = CMax (L2, keep smallest) or CMin (inner product, keep largest). */
template <class C>
void search_rowwise_int8(
        const IndexRowwiseInt8& index,
        idx_t n,
        const int8_t* qc,
        const float* qs,
        idx_t k,
        float* distances,
        idx_t* labels) {
    const size_t d = index.d;
    const size_t cs = index.code_size;
    const idx_t nb = index.ntotal;
    const bool is_l2 = index.metric_type == METRIC_L2;
    const uint8_t* codes = index.codes.data();

    // bank: scale, |b|^2 and sum of codes
    std::vector<float> bs(nb), bn(nb);
    std::vector<int32_t> bsum(nb);
#pragma omp parallel for if (nb > 1000)
    for (idx_t j = 0; j < nb; j++) {
        const int8_t* c = code_ptr(codes + j * cs);
        float s = code_scale(codes + j * cs, d);
        bs[j] = s;
        bn[j] = float(dot_i8(c, c, d)) * (s * s);
        bsum[j] = sum_i8(c, d);
    }

    // queries: |q|^2, and c_q + 128 for the VNNI kernel
    std::vector<float> qn(n);
    const bool vnni = cpu_has_vnni() && d % 64 == 0;
    std::vector<uint8_t> qu(vnni ? n * d : 0);
#pragma omp parallel for if (n > 1000)
    for (idx_t i = 0; i < n; i++) {
        const int8_t* c = qc + i * d;
        qn[i] = float(dot_i8(c, c, d)) * (qs[i] * qs[i]);
        if (vnni) {
            for (size_t j = 0; j < d; j++) {
                qu[i * d + j] = uint8_t(int(c[j]) + 128);
            }
        }
    }

    const idx_t QB = 128; // queries per task: 96 KB of codes, stays in L2
#pragma omp parallel for schedule(dynamic)
    for (idx_t q0 = 0; q0 < n; q0 += QB) {
        const idx_t nq = std::min(QB, n - q0);
        std::vector<int32_t> dots(nq);
        for (idx_t i = 0; i < nq; i++) {
            heap_heapify<C>(k, distances + (q0 + i) * k, labels + (q0 + i) * k);
        }
        for (idx_t j = 0; j < nb; j++) {
            const int8_t* b = code_ptr(codes + j * cs);
            if (vnni) {
                dots_vnni(qu.data() + q0 * d, nq, b, d, dots.data());
                const int32_t off = 128 * bsum[j];
                for (idx_t i = 0; i < nq; i++) {
                    dots[i] -= off;
                }
            } else {
                for (idx_t i = 0; i < nq; i++) {
                    dots[i] = dot_i8(qc + (q0 + i) * d, b, d);
                }
            }
            for (idx_t i = 0; i < nq; i++) {
                float ip = qs[q0 + i] * bs[j] * float(dots[i]);
                // L2 without |q|^2 (added after the selection)
                float dis = is_l2 ? bn[j] - 2.0f * ip : ip;
                float* D = distances + (q0 + i) * k;
                if (C::cmp(D[0], dis)) {
                    heap_replace_top<C>(k, D, labels + (q0 + i) * k, dis, j);
                }
            }
        }
        for (idx_t i = 0; i < nq; i++) {
            float* D = distances + (q0 + i) * k;
            idx_t* I = labels + (q0 + i) * k;
            heap_reorder<C>(k, D, I);
            if (is_l2) {
                for (idx_t r = 0; r < k; r++) {
                    if (I[r] >= 0) {
                        D[r] = std::max(D[r] + qn[q0 + i], 0.0f);
                    }
                }
            }
        }
    }
}

struct RowwiseInt8DistanceComputer : FlatCodesDistanceComputer {
    size_t d;
    bool is_l2;
    std::vector<int8_t> qc;
    float qs = 1, qn = 0;

    RowwiseInt8DistanceComputer(const IndexRowwiseInt8& index)
            : FlatCodesDistanceComputer(index.codes.data(), index.code_size),
              d(index.d),
              is_l2(index.metric_type == METRIC_L2),
              qc(index.d) {}

    void set_query(const float* x) override {
        rowwise_int8_encode(d, 1, x, qc.data(), &qs);
        qn = float(dot_i8(qc.data(), qc.data(), d)) * (qs * qs);
    }

    float distance_to_code(const uint8_t* code) override {
        const int8_t* c = code_ptr(code);
        float s = code_scale(code, d);
        float ip = qs * s * float(dot_i8(qc.data(), c, d));
        if (!is_l2) {
            return ip;
        }
        return qn + float(dot_i8(c, c, d)) * (s * s) - 2.0f * ip;
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        const uint8_t* ci = codes + i * code_size;
        const uint8_t* cj = codes + j * code_size;
        float si = code_scale(ci, d), sj = code_scale(cj, d);
        float ip = si * sj * float(dot_i8(code_ptr(ci), code_ptr(cj), d));
        if (!is_l2) {
            return ip;
        }
        float ni = float(dot_i8(code_ptr(ci), code_ptr(ci), d)) * (si * si);
        float nj = float(dot_i8(code_ptr(cj), code_ptr(cj), d)) * (sj * sj);
        return ni + nj - 2.0f * ip;
    }
};

} // namespace

void rowwise_int8_encode(
        size_t d,
        size_t n,
        const float* x,
        int8_t* codes,
        float* scales) {
#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < int64_t(n); i++) {
        const float* xi = x + i * d;
        float m = 0;
        for (size_t j = 0; j < d; j++) {
            m = std::max(m, std::fabs(xi[j]));
        }
        float s = m / 127.0f;
        if (s == 0) {
            s = 1.0f;
        }
        int8_t* ci = codes + i * d;
        for (size_t j = 0; j < d; j++) {
            float v = std::nearbyint(xi[j] / s); // round half to even
            ci[j] = int8_t(std::min(127.0f, std::max(-127.0f, v)));
        }
        scales[i] = s;
    }
}

IndexRowwiseInt8::IndexRowwiseInt8(idx_t d, MetricType metric)
        : IndexFlatCodes(d + sizeof(float), d, metric) {
    FAISS_THROW_IF_NOT_MSG(
            metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT,
            "IndexRowwiseInt8: only L2 and inner product");
    is_trained = true;
}

IndexRowwiseInt8::IndexRowwiseInt8() : IndexRowwiseInt8(0) {}

void IndexRowwiseInt8::sa_encode(idx_t n, const float* x, uint8_t* bytes)
        const {
    std::vector<int8_t> c(n * d);
    std::vector<float> s(n);
    rowwise_int8_encode(d, n, x, c.data(), s.data());
    for (idx_t i = 0; i < n; i++) {
        memcpy(bytes + i * code_size, c.data() + i * d, d);
        memcpy(bytes + i * code_size + d, &s[i], sizeof(float));
    }
}

void IndexRowwiseInt8::sa_decode(idx_t n, const uint8_t* bytes, float* x)
        const {
    for (idx_t i = 0; i < n; i++) {
        const uint8_t* code = bytes + i * code_size;
        const int8_t* c = code_ptr(code);
        float s = code_scale(code, d);
        for (idx_t j = 0; j < d; j++) {
            x[i * d + j] = float(c[j]) * s;
        }
    }
}

void IndexRowwiseInt8::add_int8(idx_t n, const int8_t* c, const float* scales) {
    if (n == 0) {
        return;
    }
    size_t old = codes.size();
    codes.resize(old + n * code_size);
    for (idx_t i = 0; i < n; i++) {
        uint8_t* code = codes.data() + old + i * code_size;
        memcpy(code, c + i * d, d);
        memcpy(code + d, &scales[i], sizeof(float));
    }
    ntotal += n;
}

void IndexRowwiseInt8::get_int8(idx_t i0, idx_t n, int8_t* c, float* scales)
        const {
    FAISS_THROW_IF_NOT(i0 >= 0 && n >= 0 && i0 + n <= ntotal);
    for (idx_t i = 0; i < n; i++) {
        const uint8_t* code = codes.data() + (i0 + i) * code_size;
        memcpy(c + i * d, code, d);
        scales[i] = code_scale(code, d);
    }
}

void IndexRowwiseInt8::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            !params || !params->sel,
            "IndexRowwiseInt8: search with an IDSelector is not supported");
    std::vector<int8_t> qc(n * d);
    std::vector<float> qs(n);
    rowwise_int8_encode(d, n, x, qc.data(), qs.data());
    search_int8(n, qc.data(), qs.data(), k, distances, labels);
}

void IndexRowwiseInt8::search_int8(
        idx_t n,
        const int8_t* qc,
        const float* qs,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    FAISS_THROW_IF_NOT(k > 0);
    if (metric_type == METRIC_L2) {
        search_rowwise_int8<CMax<float, idx_t>>(
                *this, n, qc, qs, k, distances, labels);
    } else {
        search_rowwise_int8<CMin<float, idx_t>>(
                *this, n, qc, qs, k, distances, labels);
    }
}

FlatCodesDistanceComputer* IndexRowwiseInt8::get_FlatCodesDistanceComputer()
        const {
    return new RowwiseInt8DistanceComputer(*this);
}

} // namespace faiss
