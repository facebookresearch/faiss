/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/scalar_quantizer/training.h>

#include <faiss/impl/FaissAssert.h>
#include <algorithm>
#include <cmath>

namespace faiss {

namespace scalar_quantizer {
/*******************************************************************
 * Quantizer range training
 */

static float sqr(float x) {
    return x * x;
}

constexpr size_t kTurboQuantMaxBits = 8;
// TurboQuant builds a 1-D optimal scalar quantizer analytically. We approximate
// the target density on a uniform grid over [-1, 1]; the grid is kept dense
// enough both in absolute terms and per output centroid.
constexpr size_t kTurboQuantGridMin = 1 << 15;
constexpr size_t kTurboQuantGridPerCentroid = 512;
constexpr int kTurboQuantMaxIter = 100;
constexpr double kTurboQuantTol = 1e-8;

void build_TurboQuantMSECodebook(
        size_t d,
        size_t nbits,
        std::vector<float>& centroids,
        std::vector<float>& boundaries) {
    FAISS_THROW_IF_NOT_FMT(
            nbits <= kTurboQuantMaxBits,
            "invalid TurboQuant nbits %zu (must be in [0, %zu])",
            nbits,
            kTurboQuantMaxBits);

    if (nbits == 0) {
        centroids.clear();
        boundaries.clear();
        return;
    }

    const size_t k = size_t(1) << nbits;

    if (d == 1) {
        // In 1-D, a unit vector can only be -1 or +1, so the marginal
        // distribution collapses to two atoms. The TurboQuant codebook is
        // therefore a repeated pair of endpoint centroids.
        centroids.resize(k);
        for (size_t i = 0; i < k; i++) {
            centroids[i] = i < k / 2 ? -1.0f : 1.0f;
        }
        boundaries.resize(k - 1);
        for (size_t i = 0; i + 1 < k; i++) {
            boundaries[i] = 0.5f * (centroids[i] + centroids[i + 1]);
        }
        return;
    }

    // For d > 1, TurboQuant uses the marginal distribution of one coordinate of
    // a random unit vector in R^d. On [-1, 1], this density is proportional to
    // (1 - x^2)^((d - 3) / 2), which is a symmetric beta-law after a change of
    // variables. The code below discretizes that density.
    const size_t ngrid =
            std::max(kTurboQuantGridMin, k * kTurboQuantGridPerCentroid);
    const double step = 2.0 / ngrid;
    const double alpha = 0.5 * (double(d) - 3.0);

    std::vector<double> xs(ngrid);
    // prefix_w stores the cumulative mass of the discretized density and
    // prefix_wx stores its cumulative first moment, so interval means can be
    // recovered in O(1).
    std::vector<double> prefix_w(ngrid + 1, 0.0);
    std::vector<double> prefix_wx(ngrid + 1, 0.0);

    for (size_t i = 0; i < ngrid; i++) {
        const double x = -1.0 + (i + 0.5) * step;
        const double one_minus_x2 = std::max(0.0, 1.0 - x * x);
        double w;
        if (alpha == 0.0) { // when d == 3
            w = 1.0;
        } else {
            // (1-x^2)^((d-3)/2)
            w = std::pow(one_minus_x2, alpha);
        }
        if (!std::isfinite(w) || w < 0.0) {
            w = 0.0;
        }
        xs[i] = x;
        prefix_w[i + 1] = prefix_w[i] + w;
        prefix_wx[i + 1] = prefix_wx[i] + w * x;
    }

    auto range_mean = [&](size_t i0, size_t i1, double fallback) {
        const double w = prefix_w[i1] - prefix_w[i0];
        if (w <= 0.0) {
            return fallback;
        }
        return (prefix_wx[i1] - prefix_wx[i0]) / w;
    };

    const double total_w = prefix_w.back();
    std::vector<size_t> cuts(k + 1, 0);
    cuts[k] = ngrid;

    // Initialize with k equal-mass cells under the target density. This gives
    // a stable starting point before the Lloyd refinements below.
    for (size_t i = 1; i < k; i++) {
        const double target = total_w * i / k;
        cuts[i] = std::lower_bound(prefix_w.begin(), prefix_w.end(), target) -
                prefix_w.begin();
        cuts[i] = std::min(cuts[i], ngrid);
    }

    std::vector<double> centroids_d(k);
    for (size_t i = 0; i < k; i++) {
        const double left = -1.0 + 2.0 * i / k;
        const double right = -1.0 + 2.0 * (i + 1) / k;
        // First estimate of each centroid: the conditional mean of its initial
        // equal-mass cell, with a uniform-cell midpoint as a fallback.
        centroids_d[i] = range_mean(cuts[i], cuts[i + 1], 0.5 * (left + right));
    }

    std::vector<double> boundaries_d(k > 0 ? k - 1 : 0);

    // Refine the 1-D codebook with a weighted Lloyd iteration over the
    // discretized marginal density on [-1, 1]:
    // 1. boundaries_d are the Voronoi separators implied by neighboring
    //    centroids.
    // 2. cuts map each boundary interval back to a contiguous range of the
    //    integration grid xs[].
    // 3. each centroid becomes the weighted mean of the samples currently in
    //    its cell, clipped to stay within its neighboring boundaries.
    //
    // The loop stops once the largest centroid update is below kTurboQuantTol.
    for (int iter = 0; iter < kTurboQuantMaxIter; iter++) {
        // Midpoints between adjacent centroids define the current Voronoi
        // partition of [-1, 1].
        for (size_t i = 0; i + 1 < k; i++) {
            boundaries_d[i] = 0.5 * (centroids_d[i] + centroids_d[i + 1]);
        }

        cuts[0] = 0;
        cuts[k] = ngrid;
        // Reassign the discretized density samples to the Voronoi cell induced
        // by each boundary. Because xs is sorted, the reassignment reduces to
        // finding the first grid point strictly greater than each boundary.
        for (size_t i = 1; i < k; i++) {
            cuts[i] = std::upper_bound(
                              xs.begin(), xs.end(), boundaries_d[i - 1]) -
                    xs.begin();
        }

        double max_delta = 0.0;
        for (size_t i = 0; i < k; i++) {
            const double left = i == 0 ? -1.0 : boundaries_d[i - 1];
            const double right = i + 1 == k ? 1.0 : boundaries_d[i];
            // Lloyd update: replace the centroid with the weighted average of
            // the mass assigned to its cell. Empty cells fall back to the cell
            // midpoint, and we clamp to [left, right] to preserve ordering.
            double c = range_mean(cuts[i], cuts[i + 1], 0.5 * (left + right));
            c = std::min(std::max(c, left), right);
            max_delta = std::max(max_delta, std::abs(c - centroids_d[i]));
            centroids_d[i] = c;
        }

        if (max_delta < kTurboQuantTol) {
            break;
        }
    }

    std::sort(centroids_d.begin(), centroids_d.end());

    centroids.resize(k);
    boundaries.resize(k - 1);
    for (size_t i = 0; i < k; i++) {
        centroids[i] = centroids_d[i];
    }
    for (size_t i = 0; i + 1 < k; i++) {
        boundaries[i] = 0.5f * (centroids[i] + centroids[i + 1]);
    }
}

void train_TurboQuantMSE(size_t d, size_t nbits, std::vector<float>& trained) {
    FAISS_THROW_IF_NOT_FMT(
            nbits > 0, "invalid TurboQuant SQ nbits %zu (must be > 0)", nbits);
    std::vector<float> centroids;
    std::vector<float> boundaries;
    build_TurboQuantMSECodebook(d, nbits, centroids, boundaries);
    const size_t k = centroids.size();

    trained.resize(k + (k - 1));
    for (size_t i = 0; i < k; i++) {
        trained[i] = centroids[i];
    }
    for (size_t i = 0; i + 1 < k; i++) {
        trained[k + i] = boundaries[i];
    }
}

void train_Uniform(
        RangeStat rs,
        float rs_arg,
        idx_t n,
        int k,
        const float* x,
        std::vector<float>& trained) {
    FAISS_THROW_IF_NOT(n > 0);
    trained.resize(2);
    float& vmin = trained[0];
    float& vmax = trained[1];

    if (rs == ScalarQuantizer::RS_minmax) {
        vmin = HUGE_VAL;
        vmax = -HUGE_VAL;
        for (idx_t i = 0; i < n; i++) {
            if (x[i] < vmin) {
                vmin = x[i];
            }
            if (x[i] > vmax) {
                vmax = x[i];
            }
        }
        float vexp = (vmax - vmin) * rs_arg;
        vmin -= vexp;
        vmax += vexp;
    } else if (rs == ScalarQuantizer::RS_meanstd) {
        double sum = 0, sum2 = 0;
        for (idx_t i = 0; i < n; i++) {
            sum += x[i];
            sum2 += x[i] * x[i];
        }
        float mean = sum / n;
        float var = sum2 / n - mean * mean;
        float std = var <= 0 ? 1.0 : std::sqrt(var);

        vmin = mean - std * rs_arg;
        vmax = mean + std * rs_arg;
    } else if (rs == ScalarQuantizer::RS_quantiles) {
        std::vector<float> x_copy(n);
        memcpy(x_copy.data(), x, n * sizeof(*x));
        idx_t o = static_cast<idx_t>(rs_arg * n);
        if (o < 0) {
            o = 0;
        }
        if (o > n - o) {
            o = n / 2;
        }
        std::nth_element(x_copy.begin(), x_copy.begin() + o, x_copy.end());
        vmin = x_copy[o];
        std::nth_element(
                x_copy.begin(), x_copy.begin() + (n - 1 - o), x_copy.end());
        vmax = x_copy[n - 1 - o];

    } else if (rs == ScalarQuantizer::RS_optim) {
        float a, b;
        float sx = 0;
        {
            vmin = HUGE_VAL, vmax = -HUGE_VAL;
            for (idx_t i = 0; i < n; i++) {
                if (x[i] < vmin) {
                    vmin = x[i];
                }
                if (x[i] > vmax) {
                    vmax = x[i];
                }
                sx += x[i];
            }
            b = vmin;
            a = (vmax - vmin) / (k - 1);
        }
        int verbose = false;
        int niter = 2000;
        float last_err = -1;
        int iter_last_err = 0;
        for (int it = 0; it < niter; it++) {
            float sn = 0, sn2 = 0, sxn = 0, err1 = 0;

            for (idx_t i = 0; i < n; i++) {
                float xi = x[i];
                float ni = floor((xi - b) / a + 0.5);
                if (ni < 0) {
                    ni = 0;
                }
                if (ni >= k) {
                    ni = k - 1;
                }
                err1 += sqr(xi - (ni * a + b));
                sn += ni;
                sn2 += ni * ni;
                sxn += ni * xi;
            }

            if (err1 == last_err) {
                iter_last_err++;
                if (iter_last_err == 16) {
                    break;
                }
            } else {
                last_err = err1;
                iter_last_err = 0;
            }

            float det = sqr(sn) - sn2 * n;

            b = (sn * sxn - sn2 * sx) / det;
            a = (sn * sx - n * sxn) / det;
            if (verbose) {
                printf("it %d, err1=%g            \r", it, err1);
                fflush(stdout);
            }
        }
        if (verbose) {
            printf("\n");
        }

        vmin = b;
        vmax = b + a * (k - 1);

    } else {
        FAISS_THROW_MSG("Invalid qtype");
    }
    vmax -= vmin;
}

void train_NonUniform(
        RangeStat rs,
        float rs_arg,
        idx_t n,
        int d,
        int k,
        const float* x,
        std::vector<float>& trained) {
    trained.resize(static_cast<size_t>(2) * d);
    float* vmin = trained.data();
    float* vmax = trained.data() + d;
    if (rs == ScalarQuantizer::RS_minmax) {
        memcpy(vmin, x, sizeof(*x) * d);
        memcpy(vmax, x, sizeof(*x) * d);
        for (idx_t i = 1; i < n; i++) {
            const float* xi = x + i * d;
            for (int j = 0; j < d; j++) {
                if (xi[j] < vmin[j]) {
                    vmin[j] = xi[j];
                }
                if (xi[j] > vmax[j]) {
                    vmax[j] = xi[j];
                }
            }
        }
        float* vdiff = vmax;
        for (int j = 0; j < d; j++) {
            float vexp = (vmax[j] - vmin[j]) * rs_arg;
            vmin[j] -= vexp;
            vmax[j] += vexp;
            vdiff[j] = vmax[j] - vmin[j];
        }
    } else {
        // transpose
        std::vector<float> xt(n * d);
        for (idx_t i = 1; i < n; i++) {
            const float* xi = x + i * d;
            for (int j = 0; j < d; j++) {
                xt[j * n + i] = xi[j];
            }
        }
        std::vector<float> trained_d(2);
#pragma omp parallel for
        for (int j = 0; j < d; j++) {
            train_Uniform(rs, rs_arg, n, k, xt.data() + j * n, trained_d);
            vmin[j] = trained_d[0];
            vmax[j] = trained_d[1];
        }
    }
}

} // namespace scalar_quantizer

} // namespace faiss
