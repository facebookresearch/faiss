/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <faiss/SuperKMeans.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/AdSampling.h>
#include <faiss/impl/ClusteringHelpers.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/PdxLayout.h>
#include <faiss/impl/simd_dispatch.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/simd_impl/super_kmeans_kernels.h>
#include <faiss/utils/utils.h>

#ifndef FINTEGER
#define FINTEGER long
#endif

extern "C" {

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
}

namespace faiss {

namespace {

struct TrainState {
    /// Orthogonal rotation. Train in rotated space (X_tilde = X * R);
    /// un-rotate centroids before return.
    faiss::RandomRotationMatrix R;

    std::vector<float> X_tilde; // (n, d) row-major
    int n = 0;
    std::vector<float> Y_tilde; // (k, d) row-major

    std::vector<int32_t> assignments; // size n
    std::vector<float> best_dists;    // size n; tau per vector

    int d_prime = 0;

    /// ADSampling threshold table; size d+1.
    std::vector<float> ad_coeff;

    /// Counter for the verbose-mode "low pruning" warning.
    int low_pruning_streak = 0;
    bool low_pruning_warning_printed = false;

    explicit TrainState(int d) : R(d, d) {}
};

/// PDX block layout for the trailing pruning sweep: block b covers original
/// dims [true_block_end[b] - block_dim[b], true_block_end[b]).
struct LocalBlockLayout {
    std::vector<int> block_dim;
    std::vector<int> true_block_end;
};

LocalBlockLayout build_pdx_block_layout(
        int d,
        int d_prime,
        int pdx_block_size) {
    LocalBlockLayout layout;
    const int d_trail = d - d_prime;
    const int n_full_blocks = d_trail / pdx_block_size;
    const int tail = d_trail % pdx_block_size;
    const int n_blocks = n_full_blocks + (tail > 0 ? 1 : 0);
    layout.block_dim.assign(n_blocks, pdx_block_size);
    layout.true_block_end.resize(n_blocks);
    for (int b = 0; b < n_full_blocks; ++b) {
        layout.true_block_end[b] = d_prime + (b + 1) * pdx_block_size;
    }
    if (tail > 0) {
        layout.block_dim[n_full_blocks] = tail;
        layout.true_block_end[n_full_blocks] = d;
    }
    return layout;
}

/// Iter 0: full GEMM via knn_L2sqr (vanilla Lloyd's). Fills
/// state.assignments and state.best_dists. Returns objective.
double run_iter0_full_gemm(int d, int k, TrainState& state) {
    std::vector<int64_t> labels(state.n);
    std::vector<float> distances(state.n);
    knn_L2sqr(
            state.X_tilde.data(),
            state.Y_tilde.data(),
            d,
            state.n,
            k,
            /*k=*/1,
            distances.data(),
            labels.data(),
            /*y_norm2=*/nullptr);

    assert(!state.assignments.empty());
    assert(!state.best_dists.empty());
    double objective = 0.0;
    for (int i = 0; i < state.n; ++i) {
        state.assignments[i] = static_cast<int>(labels[i]);
        state.best_dists[i] = distances[i];
        objective += distances[i];
    }
    return objective;
}

/// Iter 1+: refresh tau, then run one shared super_kmeans_assign_iteration
/// pass (partial GEMM + ADSampling progressive pruning). Updates
/// state.assignments and state.best_dists, writes total_pairs and
/// pruned_at_gemm, and returns the objective.
double run_iter_pruned(
        int d,
        int k,
        const SuperKMeansParameters& cp,
        TrainState& state,
        SuperKMeansAssignScratch& assign_scratch,
        int64_t& total_pairs,
        int64_t& pruned_at_gemm) {
    const int n_train = state.n;
    assert(state.d_prime >= 1);
    assert(!state.ad_coeff.empty());
    assert(static_cast<int>(state.best_dists.size()) >= n_train);
    assert(static_cast<int>(state.assignments.size()) >= n_train);

    // Refresh tau: exact full-d L2 distance to the currently assigned centroid.
    // super_kmeans_assign_iteration requires an exact tau (the chi-squared
    // bound assumes it), and centroids moved since the previous assignment.
#pragma omp parallel for
    for (int i = 0; i < n_train; ++i) {
        const int j_prev = state.assignments[i];
        const float* xrow = state.X_tilde.data() + static_cast<size_t>(i) * d;
        const float* yrow =
                state.Y_tilde.data() + static_cast<size_t>(j_prev) * d;
        float tau = 0.0f;
        for (int m = 0; m < d; ++m) {
            const float diff = xrow[m] - yrow[m];
            tau += diff * diff;
        }
        state.best_dists[i] = tau;
    }

    super_kmeans_assign_iteration(
            state.X_tilde.data(),
            n_train,
            d,
            state.Y_tilde.data(),
            k,
            state.best_dists.data(),
            state.assignments.data(),
            state.d_prime,
            state.ad_coeff.data(),
            cp,
            &total_pairs,
            &pruned_at_gemm,
            &assign_scratch);

    double objective = 0.0;
    for (int i = 0; i < n_train; ++i) {
        objective += state.best_dists[i];
    }
    return objective;
}

/// Post-iteration: update centroids and split empties. Returns nsplit.
int update_centroids_and_split(
        int d,
        int k,
        TrainState& state,
        std::vector<int64_t>& labels64,
        std::vector<float>& hassign) {
    std::fill(hassign.begin(), hassign.end(), 0.0f);
    assert(!labels64.empty());
    assert(!state.assignments.empty());
    for (int i = 0; i < state.n; ++i) {
        labels64[i] = static_cast<int64_t>(state.assignments[i]);
    }
    detail::compute_centroids(
            d,
            k,
            state.n,
            /*k_frozen=*/0,
            reinterpret_cast<const uint8_t*>(state.X_tilde.data()),
            /*codec=*/nullptr,
            labels64.data(),
            /*weights=*/nullptr,
            hassign.data(),
            state.Y_tilde.data());
    if (state.n <= k) {
        return 0;
    }
    return detail::split_clusters(
            d,
            k,
            state.n,
            /*k_frozen=*/0,
            hassign.data(),
            state.Y_tilde.data());
}

/// Stay-in-band controller: nudge state.d_prime based on observed pruning
/// rate. Returns the observed pruning rate (0 when there were no pairs).
float adapt_d_prime(
        int d,
        const SuperKMeansParameters& cp,
        TrainState& state,
        int64_t total_pairs,
        int64_t pruned_at_gemm) {
    if (total_pairs == 0) {
        return 0.0f;
    }
    const float pruning_rate = static_cast<float>(pruned_at_gemm) /
            static_cast<float>(total_pairs);
    int new_dp = state.d_prime;
    if (pruning_rate > cp.pruning_target_high) {
        new_dp = static_cast<int>(
                std::lround(state.d_prime * (1.0f - cp.d_prime_adjust)));
    } else if (pruning_rate < cp.pruning_target_low) {
        new_dp = static_cast<int>(
                std::lround(state.d_prime * (1.0f + cp.d_prime_adjust)));
    }
    new_dp = std::max(cp.d_prime_min, new_dp);
    new_dp = std::min(d / 2, new_dp);
    state.d_prime = new_dp;
    return pruning_rate;
}

/// Pre-loop setup: subsample, rotate, Forgy init, build ADSampling table,
/// size the label scratch. Returned `sampled_x_owner` keeps the subsampled
/// buffer alive when subsampling occurred (otherwise empty).
std::unique_ptr<uint8_t[]> setup_train_state(
        TrainState& state,
        std::vector<int64_t>& labels64,
        std::vector<float>& hassign,
        const SuperKMeansParameters& cp,
        int d,
        int k,
        idx_t n,
        const float* x) {
    const size_t line_size = sizeof(float) * static_cast<size_t>(d);
    idx_t nx = n;
    const uint8_t* x_bytes = reinterpret_cast<const uint8_t*>(x);
    std::unique_ptr<uint8_t[]> sampled_x_owner;
    if (static_cast<size_t>(nx) >
        static_cast<size_t>(k) * cp.max_points_per_centroid) {
        Clustering tmp_clus(d, k, cp);
        uint8_t* x_new = nullptr;
        float* w_unused = nullptr;
        nx = detail::subsample_training_set(
                tmp_clus,
                nx,
                x_bytes,
                line_size,
                /*weights=*/nullptr,
                &x_new,
                &w_unused);
        FAISS_ASSERT(x_new != nullptr);
        sampled_x_owner.reset(x_new);
        x_bytes = x_new;
    }
    const float* x_sampled = reinterpret_cast<const float*>(x_bytes);

    FAISS_THROW_IF_NOT_MSG(
            nx <= static_cast<idx_t>(std::numeric_limits<int>::max()),
            "SuperKMeans: training set size exceeds INT_MAX after sampling");
    state.n = static_cast<int>(nx);

    state.R.init(cp.seed);

    state.X_tilde.resize(static_cast<size_t>(state.n) * d);
    state.R.apply_noalloc(state.n, x_sampled, state.X_tilde.data());

    // Forgy init: pick k random rows from the rotated pool as initial
    // centroids. These remain in rotated space; un-rotation happens
    // after the iteration loop.
    state.Y_tilde.resize(static_cast<size_t>(k) * d);
    {
        std::vector<int> perm(state.n);
        rand_perm(perm.data(), state.n, static_cast<int64_t>(cp.seed) + 1);
        for (int j = 0; j < k; ++j) {
            std::memcpy(
                    state.Y_tilde.data() + static_cast<size_t>(j) * d,
                    state.X_tilde.data() + static_cast<size_t>(perm[j]) * d,
                    sizeof(float) * d);
        }
    }

    state.d_prime =
            std::max(cp.d_prime_min, static_cast<int>(d * cp.d_prime_fraction));
    state.d_prime = std::min(state.d_prime, d / 2);

    const double epsilon = static_cast<double>(cp.ad_epsilon_factor) / d;
    state.ad_coeff = detail::precompute_ad_thresholds(d, epsilon);
    FAISS_ASSERT_MSG(
            state.ad_coeff.size() == static_cast<size_t>(d + 1),
            "ad_coeff size mismatch");

    state.assignments.assign(state.n, 0);
    state.best_dists.assign(state.n, std::numeric_limits<float>::max());

    hassign.assign(k, 0.0f);

    labels64.resize(state.n);

    return sampled_x_owner;
}

/// Un-rotate centroids into output buffer. R orthogonal, so
/// reverse_transform applies R^T = R^-1.
void untransform_centroids(
        std::vector<float>& centroids,
        const RandomRotationMatrix& R,
        int d,
        int k,
        const float* Y_tilde) {
    centroids.resize(static_cast<size_t>(k) * d);
    R.reverse_transform(k, Y_tilde, centroids.data());
}

} // namespace

SuperKMeans::SuperKMeans(int d, int k, const SuperKMeansParameters& cp_in)
        : cp(cp_in), d(d), k(k) {
    FAISS_THROW_IF_NOT_MSG(d > 0, "SuperKMeans: d must be positive");
    FAISS_THROW_IF_NOT_MSG(k > 0, "SuperKMeans: k must be positive");
    FAISS_THROW_IF_NOT_MSG(
            cp.d_prime_fraction > 0.0f && cp.d_prime_fraction <= 1.0f,
            "SuperKMeans: d_prime_fraction must be in (0, 1]");
    FAISS_THROW_IF_NOT_MSG(
            cp.d_prime_adjust >= 0.0f && cp.d_prime_adjust < 1.0f,
            "SuperKMeans: d_prime_adjust must be in [0, 1)");
    // d >= 2 * d_prime_min keeps d_prime in the chi-squared validity
    // range after both clamping steps (floor at d_prime_min, ceiling
    // at d/2). See AdSampling.h on the p >= 16 contract.
    FAISS_THROW_IF_NOT_FMT(
            d >= 2 * cp.d_prime_min,
            "SuperKMeans: d (%d) must be >= 2 * d_prime_min (%d)",
            d,
            cp.d_prime_min);
    FAISS_THROW_IF_NOT_MSG(
            cp.d_prime_min >= 16,
            "SuperKMeans: d_prime_min must be >= 16 (chi-squared validity floor)");
    FAISS_THROW_IF_NOT_MSG(
            cp.pdx_block_size > 0, "SuperKMeans: pdx_block_size must be > 0");
    FAISS_THROW_IF_NOT_MSG(cp.x_batch > 0, "SuperKMeans: x_batch must be > 0");
    FAISS_THROW_IF_NOT_MSG(cp.y_batch > 0, "SuperKMeans: y_batch must be > 0");
    FAISS_THROW_IF_NOT_MSG(
            cp.pruning_target_low > 0.0f &&
                    cp.pruning_target_low <= cp.pruning_target_high &&
                    cp.pruning_target_high < 1.0f,
            "SuperKMeans: require 0 < pruning_target_low <= pruning_target_high < 1");
    // epsilon = ad_epsilon_factor / d is the chi-squared significance
    // level. precompute_ad_thresholds requires epsilon in (0, 1).
    FAISS_THROW_IF_NOT_MSG(
            cp.ad_epsilon_factor > 0.0f &&
                    cp.ad_epsilon_factor < static_cast<float>(d),
            "SuperKMeans: ad_epsilon_factor must be in (0, d) "
            "so epsilon = factor/d is in (0,1)");
    FAISS_THROW_IF_NOT_MSG(
            cp.omp_chunk > 0, "SuperKMeans: omp_chunk must be > 0");
}

void SuperKMeans::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(n > 0, "SuperKMeans: n must be positive");
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "SuperKMeans: x must not be null");
    FAISS_THROW_IF_NOT_MSG(
            n >= static_cast<idx_t>(k), "SuperKMeans: n must be >= k");
    if (cp.check_input_data_for_NaNs) {
        for (size_t i = 0; i < static_cast<size_t>(n) * d; i++) {
            FAISS_THROW_IF_NOT_MSG(
                    std::isfinite(x[i]),
                    "SuperKMeans: input contains NaN's or Inf's");
        }
    }
    if (cp.verbose && n < static_cast<idx_t>(k) * cp.min_points_per_centroid) {
        printf("WARNING: clustering %" PRId64
               " points to %d centroids: please provide at least "
               "%" PRId64 " training points\n",
               n,
               k,
               static_cast<idx_t>(k) * cp.min_points_per_centroid);
    }

    TrainState state(d);
    std::vector<int64_t> labels64;
    SuperKMeansAssignScratch assign_scratch;
    std::vector<float> hassign;
    [[maybe_unused]] auto sampled_x_owner =
            setup_train_state(state, labels64, hassign, cp, d, k, n, x);

    iteration_stats.clear();
    iteration_stats.reserve(cp.niter);
    gemm_pruning_rates.clear();
    gemm_pruning_rates.reserve(cp.niter);

    const double t_train_start = getmillisecs();

    for (int iter = 0; iter < cp.niter; ++iter) {
        const double t_iter_start = getmillisecs();
        double objective = 0.0;
        int64_t total_pairs = 0;
        int64_t pruned_at_gemm = 0;

        if (iter == 0) {
            objective = run_iter0_full_gemm(d, k, state);
        } else {
            objective = run_iter_pruned(
                    d,
                    k,
                    cp,
                    state,
                    assign_scratch,
                    total_pairs,
                    pruned_at_gemm);
        }

        const int nsplit =
                update_centroids_and_split(d, k, state, labels64, hassign);
        const float pruning_rate = (iter == 0)
                ? 0.0f
                : adapt_d_prime(d, cp, state, total_pairs, pruned_at_gemm);

        ClusteringIterationStats stat{};
        stat.obj = static_cast<float>(objective);
        stat.time = (getmillisecs() - t_iter_start) / 1000.0;
        stat.time_search = stat.time;
        stat.imbalance_factor = std::numeric_limits<double>::quiet_NaN();
        stat.nsplit = nsplit;
        iteration_stats.push_back(stat);
        gemm_pruning_rates.push_back(pruning_rate);

        if (iter > 0) {
            if (pruning_rate < 0.85f) {
                state.low_pruning_streak++;
            } else {
                state.low_pruning_streak = 0;
            }
            if (cp.verbose && state.low_pruning_streak >= 3 &&
                !state.low_pruning_warning_printed) {
                fprintf(stderr,
                        "WARNING: SuperKMeans steady-state pruning < 0.85 for 3+ iters "
                        "(current=%.2f). Data may not be a good fit for ADSampling; "
                        "consider falling back to faiss::Clustering.\n",
                        pruning_rate);
                state.low_pruning_warning_printed = true;
            }
        }

        if (cp.verbose) {
            printf("  Iter %d: obj=%g time=%.3fs prune=%.4f dp=%d nsplit=%d\n",
                   iter,
                   stat.obj,
                   stat.time,
                   pruning_rate,
                   state.d_prime,
                   nsplit);
        }
    }

    if (cp.verbose) {
        printf("Total training time: %.3fs\n",
               (getmillisecs() - t_train_start) / 1000.0);
    }

    untransform_centroids(centroids, state.R, d, k, state.Y_tilde.data());
}

void super_kmeans_assign_iteration(
        const float* X_tilde,
        int n,
        int d,
        const float* Y_tilde,
        int k,
        float* tau,
        int32_t* assignments,
        int d_prime,
        const float* ad_coeff,
        const SuperKMeansParameters& cp,
        int64_t* total_pairs_out,
        int64_t* pruned_at_gemm_out,
        SuperKMeansAssignScratch* scratch_in) {
    FAISS_THROW_IF_NOT_MSG(
            d_prime >= 1,
            "super_kmeans_assign_iteration: d_prime must be >= 1");
    FAISS_THROW_IF_NOT_MSG(
            d_prime < d, "super_kmeans_assign_iteration: d_prime must be < d");
    FAISS_THROW_IF_NOT_MSG(
            ad_coeff != nullptr,
            "super_kmeans_assign_iteration: ad_coeff must not be null");
    FAISS_THROW_IF_NOT_MSG(
            tau != nullptr,
            "super_kmeans_assign_iteration: tau must not be null");
    FAISS_THROW_IF_NOT_MSG(
            assignments != nullptr,
            "super_kmeans_assign_iteration: assignments must not be null");

    const int d_trail = d - d_prime;
    const auto layout = build_pdx_block_layout(d, d_prime, cp.pdx_block_size);
    const int n_blocks = static_cast<int>(layout.block_dim.size());

    // Reuse the caller's scratch when provided (grow-only buffers, so a caller
    // looping over iterations allocates nothing in steady state); otherwise use
    // a local instance for one-shot callers.
    SuperKMeansAssignScratch local_scratch;
    SuperKMeansAssignScratch& s = scratch_in ? *scratch_in : local_scratch;

    // PDX-lay out the centroid trailing block once per call.
    auto& Y_trail = s.Y_trail;
    auto& Y_pdx = s.Y_pdx;
    Y_trail.resize(static_cast<size_t>(k) * d_trail);
    Y_pdx.resize(static_cast<size_t>(k) * d_trail);
    for (int j = 0; j < k; ++j) {
        std::memcpy(
                Y_trail.data() + static_cast<size_t>(j) * d_trail,
                Y_tilde + static_cast<size_t>(j) * d + d_prime,
                d_trail * sizeof(float));
    }
    detail::pdxify(Y_trail.data(), k, d_trail, cp.pdx_block_size, Y_pdx.data());

    auto& x_norms_partial = s.x_norms_partial;
    auto& y_norms_partial = s.y_norms_partial;
    x_norms_partial.resize(n);
    detail::compute_partial_norms(
            X_tilde, n, d, d_prime, x_norms_partial.data());
    y_norms_partial.resize(k);
    detail::compute_partial_norms(
            Y_tilde, k, d, d_prime, y_norms_partial.data());

    int64_t total_pairs = 0;
    int64_t pruned_at_gemm = 0;

    // Size the GEMM tile buffer to the actual max tile, not cp.x_batch *
    // cp.y_batch, which over-allocates when n < x_batch or k < y_batch.
    const int bx_max = std::min(cp.x_batch, n);
    const int by_max = std::min(cp.y_batch, k);
    auto& partial_ip = s.partial_ip;
    partial_ip.resize(static_cast<size_t>(bx_max) * by_max);

    for (int xi = 0; xi < n; xi += cp.x_batch) {
        const int bx = std::min(cp.x_batch, n - xi);
        for (int yj = 0; yj < k; yj += cp.y_batch) {
            const int by = std::min(cp.y_batch, k - yj);

            // Partial GEMM over [0, d_prime): partial_ip[i*by + j] =
            //   <X[xi+i, 0:d_prime], Y[yj+j, 0:d_prime]>.
            {
                FINTEGER M = by;
                FINTEGER N_ = bx;
                FINTEGER K_ = d_prime;
                float alpha = 1.0f;
                float beta = 0.0f;
                FINTEGER lda_y = d;
                FINTEGER lda_x = d;
                FINTEGER ldc = by;
                sgemm_("Transpose",
                       "Not transpose",
                       &M,
                       &N_,
                       &K_,
                       &alpha,
                       Y_tilde + static_cast<size_t>(yj) * d,
                       &lda_y,
                       X_tilde + static_cast<size_t>(xi) * d,
                       &lda_x,
                       &beta,
                       partial_ip.data(),
                       &ldc);
            }

            // One SIMD dispatch per (xi, yj) tile.
            with_simd_level([&]<SIMDLevel SL>() {
                [[maybe_unused]] const int omp_chunk_local = cp.omp_chunk;
                int64_t tile_total = 0;
                int64_t tile_pruned = 0;
#pragma omp parallel for schedule(dynamic, omp_chunk_local) \
        reduction(+ : tile_total) reduction(+ : tile_pruned)
                for (int i = 0; i < bx; ++i) {
                    const float xnp_i = x_norms_partial[xi + i];
                    float tau_i = tau[xi + i];
                    int32_t best_j = assignments[xi + i];
                    const float* xrow =
                            X_tilde + static_cast<size_t>(xi + i) * d;
                    for (int j = 0; j < by; ++j) {
                        ++tile_total;
                        // L2-from-IP; clamp against catastrophic cancellation
                        // when the true distance is ~0.
                        float pd = xnp_i + y_norms_partial[yj + j] -
                                2.0f *
                                        partial_ip
                                                [static_cast<size_t>(i) * by +
                                                 j];
                        if (pd < 0.0f) {
                            pd = 0.0f;
                        }
                        // ADSampling chi-squared bound at the d_prime boundary.
                        if (pd > ad_coeff[d_prime] * tau_i) {
                            ++tile_pruned;
                            continue;
                        }
                        // Progressive pruning across PDX blocks; double
                        // accumulator mitigates float drift over many block
                        // additions.
                        double dist = pd;
                        bool keep = true;
                        size_t pdx_offset = 0;
                        for (int b = 0; b < n_blocks; ++b) {
                            const int n_in_block = layout.block_dim[b];
                            const int true_end = layout.true_block_end[b];
                            const float* xblk = xrow + (true_end - n_in_block);
                            const float* yblk = Y_pdx.data() + pdx_offset +
                                    static_cast<size_t>(yj + j) * n_in_block;
                            dist += faiss::detail::block_l2<SL>(
                                    xblk, yblk, n_in_block);
                            pdx_offset += static_cast<size_t>(k) * n_in_block;
                            if (dist > static_cast<double>(ad_coeff[true_end]) *
                                        tau_i) {
                                keep = false;
                                break;
                            }
                        }
                        if (keep && dist < tau_i) {
                            tau_i = static_cast<float>(dist);
                            best_j = static_cast<int32_t>(yj + j);
                        }
                    }
                    tau[xi + i] = tau_i;
                    assignments[xi + i] = best_j;
                }
                total_pairs += tile_total;
                pruned_at_gemm += tile_pruned;
            });
        }
    }
    if (total_pairs_out) {
        *total_pairs_out = total_pairs;
    }
    if (pruned_at_gemm_out) {
        *pruned_at_gemm_out = pruned_at_gemm;
    }
}

} // namespace faiss
