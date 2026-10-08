/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/pipnn/kernels.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>

extern "C" {

// this is to keep the clang syntax checker happy
#ifndef FINTEGER
#define FINTEGER int
#endif

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);

int ssyrk_(
        const char* uplo,
        const char* trans,
        FINTEGER* n,
        FINTEGER* k,
        float* alpha,
        float* a,
        FINTEGER* lda,
        float* beta,
        float* c,
        FINTEGER* ldc);
}

namespace faiss {
namespace pipnn {

namespace {

FINTEGER to_finteger(size_t v, const char* what) {
    FAISS_THROW_IF_NOT_FMT(
            v <= size_t(std::numeric_limits<FINTEGER>::max()),
            "PiPNN kernel: %s=%zu exceeds the BLAS integer range",
            what,
            v);
    return static_cast<FINTEGER>(v);
}

float nan_to_inf(float v) {
    return std::isnan(v) ? std::numeric_limits<float>::infinity() : v;
}

/// Top-k in (dist, index) order on a Faiss max-heap. Empty slots are
/// (+inf, INT32_MAX), so +inf offers still fill them, in index order.
using TopK = CMax<float, int32_t>;
constexpr int32_t kEmpty = std::numeric_limits<int32_t>::max();

void topk_init(float* D, int32_t* I, size_t k) {
    std::fill(D, D + k, std::numeric_limits<float>::infinity());
    std::fill(I, I + k, kEmpty);
}

void topk_offer(float* D, int32_t* I, size_t k, float v, int32_t id) {
    if (TopK::cmp2(D[0], v, I[0], id)) {
        heap_replace_top<TopK>(k, D, I, v, id);
    }
}

/// Sorts the slots by (dist, index); slots never filled become (+inf, -1).
void topk_finish(float* D, int32_t* I, size_t k) {
    heap_reorder<TopK>(k, D, I);
    std::replace(I, I + k, kEmpty, int32_t(-1));
}

} // namespace

void stripe_assign(
        const float* points,
        size_t S,
        const float* leaders,
        const float* leader_norms,
        size_t L,
        size_t d,
        Ranking ranking,
        int fanout,
        float* scratch,
        uint16_t* out) {
    FAISS_THROW_IF_NOT_FMT(
            L >= 1 && L <= 65535,
            "stripe_assign: L=%zu must be in [1, 65535]",
            L);
    FAISS_THROW_IF_NOT_FMT(
            fanout >= 1 && size_t(fanout) <= L,
            "stripe_assign: fanout=%d must be in [1, L=%zu]",
            fanout,
            L);
    FAISS_THROW_IF_NOT_MSG(d >= 1, "stripe_assign: d must be >= 1");
    FAISS_THROW_IF_NOT_MSG(
            ranking == Ranking::IP || leader_norms,
            "stripe_assign: leader_norms is required for the L2 and Angle rankings");
    if (S == 0) {
        return;
    }
    FINTEGER li = to_finteger(L, "L");
    FINTEGER si = to_finteger(S, "S");
    FINTEGER di = to_finteger(d, "d");

    // Column-major BLAS sees row-major scratch (S x L) as L x S, so "T", "N"
    // gives scratch[i * L + j] = alpha * p_i.l_j + beta * scratch[i * L + j].
    float alpha = -1;
    float beta = 0; // BLAS does not read C when beta == 0
    if (ranking == Ranking::L2) {
        for (size_t i = 0; i < S; i++) {
            std::copy(leader_norms, leader_norms + L, scratch + i * L);
        }
        alpha = -2;
        beta = 1;
    }
    sgemm_("T",
           "N",
           &li,
           &si,
           &di,
           &alpha,
           leaders,
           &di,
           points,
           &di,
           &beta,
           scratch,
           &li);

    if (ranking == Ranking::Angle) {
        // A zero-norm leader has no direction: rank it +inf, not the -inf of
        // a positive dot product divided by 0.
        for (size_t i = 0; i < S; i++) {
            float* row = scratch + i * L;
            for (size_t j = 0; j < L; j++) {
                const float len = std::sqrt(leader_norms[j]);
                row[j] = len > 0 ? row[j] / len
                                 : std::numeric_limits<float>::infinity();
            }
        }
    }

    const size_t f = size_t(fanout);
    std::vector<float> D(f);
    std::vector<int32_t> I(f);
    for (size_t i = 0; i < S; i++) {
        topk_init(D.data(), I.data(), f);
        const float* row = scratch + i * L;
        for (size_t j = 0; j < L; j++) {
            topk_offer(D.data(), I.data(), f, nan_to_inf(row[j]), int32_t(j));
        }
        topk_finish(D.data(), I.data(), f);
        // L >= fanout and every offer precedes an empty slot, so all fanout
        // slots are filled, even when the whole row is +inf.
        for (size_t r = 0; r < f; r++) {
            out[i * f + r] = uint16_t(I[r]);
        }
    }
}

namespace {

/// Distance from a Gram entry: L2 max(0, ni + nj - 2g), IP -g. NaN is mapped
/// to +inf before the L2 clamp, which would otherwise turn it into 0.
float gram_to_distance(bool is_l2, float ni, float nj, float g) {
    if (is_l2) {
        const float v = ni + nj - 2 * g;
        if (std::isnan(v)) {
            return std::numeric_limits<float>::infinity();
        }
        return v < 0 ? 0 : v;
    }
    return nan_to_inf(-g);
}

/// Lower triangle of X X^T, diagonal included, into row-major gram (s x s);
/// the strict upper triangle is not written.
void gram_lower(const float* X, size_t s, size_t d, float* gram) {
    // Column-major BLAS reads X as A = X^T (d x s); the upper triangle of
    // C = A^T A ("U", "T") is the row-major lower triangle of gram.
    FINTEGER si = to_finteger(s, "s");
    FINTEGER di = to_finteger(d, "d");
    float one = 1;
    float zero = 0; // BLAS does not read C when beta == 0
    ssyrk_("U",
           "T",
           &si,
           &di,
           &one,
           const_cast<float*>(X),
           &di,
           &zero,
           gram,
           &si);
}

} // namespace

void leaf_knn(
        const float* X,
        const float* norms,
        size_t s,
        size_t d,
        MetricType metric,
        int k,
        float* gram,
        int32_t* nn_idx,
        float* nn_dist) {
    FAISS_THROW_IF_NOT_MSG(
            metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT,
            "leaf_knn: metric must be METRIC_L2 or METRIC_INNER_PRODUCT");
    FAISS_THROW_IF_NOT_MSG(k >= 1, "leaf_knn: k must be >= 1");
    FAISS_THROW_IF_NOT_MSG(d >= 1, "leaf_knn: d must be >= 1");
    FAISS_THROW_IF_NOT_MSG(
            metric == METRIC_INNER_PRODUCT || norms,
            "leaf_knn: norms is required for METRIC_L2");
    const size_t kk = size_t(k);
    for (size_t i = 0; i < s; i++) {
        topk_init(nn_dist + i * kk, nn_idx + i * kk, kk);
    }
    if (s >= 2) {
        gram_lower(X, s, d, gram);
    }
    const bool is_l2 = metric == METRIC_L2;
    for (size_t i = 1; i < s; i++) {
        const float* gi = gram + i * s;
        const float ni = is_l2 ? norms[i] : 0;
        for (size_t j = 0; j < i; j++) {
            const float nj = is_l2 ? norms[j] : 0;
            const float v = gram_to_distance(is_l2, ni, nj, gi[j]);
            topk_offer(nn_dist + i * kk, nn_idx + i * kk, kk, v, int32_t(j));
            topk_offer(nn_dist + j * kk, nn_idx + j * kk, kk, v, int32_t(i));
        }
    }
    for (size_t i = 0; i < s; i++) {
        topk_finish(nn_dist + i * kk, nn_idx + i * kk, kk);
    }
}

void pairwise_distances(
        const float* X,
        const float* norms,
        size_t n,
        size_t d,
        MetricType metric,
        float* out) {
    FAISS_THROW_IF_NOT_MSG(
            metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT,
            "pairwise_distances: metric must be METRIC_L2 or METRIC_INNER_PRODUCT");
    FAISS_THROW_IF_NOT_MSG(d >= 1, "pairwise_distances: d must be >= 1");
    FAISS_THROW_IF_NOT_MSG(
            metric == METRIC_INNER_PRODUCT || norms,
            "pairwise_distances: norms is required for METRIC_L2");
    if (n == 0) {
        return;
    }
    // Row i reads only its lower part of the Gram scratch; mirrored writes of
    // earlier rows go to their strict upper triangle.
    gram_lower(X, n, d, out);
    const bool is_l2 = metric == METRIC_L2;
    for (size_t i = 0; i < n; i++) {
        float* oi = out + i * n;
        const float ni = is_l2 ? norms[i] : 0;
        for (size_t j = 0; j < i; j++) {
            const float nj = is_l2 ? norms[j] : 0;
            const float v = gram_to_distance(is_l2, ni, nj, oi[j]);
            oi[j] = v;
            out[j * n + i] = v;
        }
        oi[i] = gram_to_distance(is_l2, ni, ni, oi[i]);
    }
}

} // namespace pipnn
} // namespace faiss
